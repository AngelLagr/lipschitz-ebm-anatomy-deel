#######################################
# ## TRAIN EBM USING IMAGE DATASET ## #
#######################################

import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
import csv
import numpy as np
import random
from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip
from utils import download_flowers_data, plot_ims, plot_diagnostics
from utilfid import load_mnist_feature_extractor, compute_fid_batch

# directory for experiment results
EXP_DIR = './out_data/Expe/mnist_Lip0_Langevin100_Seed42/'
# json file with experiment config
CONFIG_FILE = './config_locker/mnist_Lip0_Langevin100_Seed42.json'


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    choice = input('Experiment folder "{}" already exists. Do you want to overwrite it? (y/n) '.format(EXP_DIR))
    if choice.lower() == 'y':
        import shutil
        shutil.rmtree(EXP_DIR)
        os.makedirs(EXP_DIR)
        for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
            os.mkdir(EXP_DIR + folder)

    else :
        raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_data.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
start_iter = 0
loaded_checkpoint = None

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Initialize FID feature extractor for MNIST
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
else:
    feature_extractor = None


########################
# ## TRAINING SETUP # ##
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip}
if config['net_type'] == "vanillaLip":
    f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
else :
    f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)

# set up optimizer
optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
# make tensor of training data
if config['data'] == 'flowers':
    download_flowers_data()
data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}
transform = tr.Compose([tr.Resize(config['im_sz']),
                        tr.CenterCrop(config['im_sz']),
                        tr.ToTensor(),
                        tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), tuple(0.5*t.ones(config['im_ch'])))])
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

# initialize persistent images from noise (one persistent image for each data image)
# s_t_0 is used when init_type == 'persistent' in sample_s_t()
s_t_0 = 2 * t.rand_like(q) - 1
if isinstance(loaded_checkpoint, dict) and ('s_t_0' in loaded_checkpoint):
    s_t_0.copy_(loaded_checkpoint['s_t_0'].to(device))


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# sample batch from given array of images
def sample_image_set(image_set, batch_size = config['batch_size']):
    rand_inds = t.randperm(image_set.shape[0])[0:batch_size]
    return image_set[rand_inds], rand_inds

# sample positive images from dataset distribution q (add noise to ensure min sd is at least langevin noise sd)
def sample_q(batch_size = config['batch_size']):
    x_q = sample_image_set(q, batch_size=batch_size)[0]
    return x_q + config['data_epsilon'] * t.randn_like(x_q)

# initialize and update images with langevin dynamics to obtain samples from finite-step MCMC distribution s_t
def sample_s_t(L, init_type, update_s_t_0=True, batch_size=config['batch_size']):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", or "gaussian")
    def sample_s_t_0():
        if init_type == 'persistent':
            # 95% chance each sample comes from s_t_0, 5% chance from uniform random noise
            mask = t.rand(batch_size) < 0.95
            indices = t.randperm(s_t_0.shape[0])[:batch_size]
            persistent_samples = s_t_0[indices]
            random_samples = 2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device) - 1
            x_s_t_0 = t.where(mask.view(-1, 1, 1, 1).to(device), persistent_samples, random_samples)
            sample_inds = indices.clone()
            #sample_inds[~mask] = -1 # mark non-persistent samples with -1 since they won't be used to update s_t_0
            return x_s_t_0, sample_inds
        elif init_type == 'data':
            return sample_q(batch_size=batch_size), None
        elif init_type == 'uniform':
            noise_image = 2 * t.rand([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]) - 1
            return noise_image.to(device), None
        elif init_type == 'gaussian':
            noise_image = t.randn([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']])
            return noise_image.to(device), None
        else:
            raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    # initialize MCMC samples
    x_s_t_0, s_t_0_inds = sample_s_t_0()

    # iterative langevin updates of MCMC samples
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).to(device)  # variable r_s_t (Section 3.2) to record average gradient magnitude
    for ell in range(L):
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t.data += - f_prime + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

    if init_type == 'persistent' and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records (see Section 3)
d_s_t_record = t.zeros(config['num_train_iters']).to(device)  # energy difference between positive and negative samples
r_s_t_record = t.zeros(config['num_train_iters']).to(device)  # average image gradient magnitude along Langevin path
fid_record = t.zeros(config['num_train_iters']).to(device)  # FID score
fid_record[:] = float('nan')
instability_record = t.zeros(config['num_train_iters']).to(device)  # stability index in [0, 1]
instability_record[:] = float('nan')
fid_score = float('nan')
f_x_q_record = t.zeros(config['num_train_iters']).to(device)  # f(x_q).mean()
f_x_st_record = t.zeros(config['num_train_iters']).to(device)  # f(x_s_t).mean()
csv_file_path = EXP_DIR + 'plots/' + 'training_metrics.csv'
last_csv_written = 0
stability_window = 2

if isinstance(loaded_checkpoint, dict):
    ckpt_iter = int(loaded_checkpoint.get('iter', 0))
    d_s_t_record[:ckpt_iter] = loaded_checkpoint['d_s_t_record'][:ckpt_iter].to(device)
    r_s_t_record[:ckpt_iter] = loaded_checkpoint['r_s_t_record'][:ckpt_iter].to(device)
    fid_record[:ckpt_iter] = loaded_checkpoint['fid_record'][:ckpt_iter].to(device)
    instability_record[:ckpt_iter] = loaded_checkpoint['instability_record'][:ckpt_iter].to(device)
    f_x_q_record[:ckpt_iter] = loaded_checkpoint['f_x_q_record'][:ckpt_iter].to(device)
    f_x_st_record[:ckpt_iter] = loaded_checkpoint['f_x_st_record'][:ckpt_iter].to(device)
    fid_score = float(loaded_checkpoint.get('fid_score', float('nan')))
    last_csv_written = int(loaded_checkpoint.get('last_csv_written', start_iter))


def save_full_checkpoint(iteration_idx):
    iter_done = iteration_idx + 1
    checkpoint = {
        'iter': iter_done,
        'model_state_dict': f.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        's_t_0': s_t_0.detach().cpu(),
        'd_s_t_record': d_s_t_record[:iter_done].detach().cpu(),
        'r_s_t_record': r_s_t_record[:iter_done].detach().cpu(),
        'fid_record': fid_record[:iter_done].detach().cpu(),
        'instability_record': instability_record[:iter_done].detach().cpu(),
        'f_x_q_record': f_x_q_record[:iter_done].detach().cpu(),
        'f_x_st_record': f_x_st_record[:iter_done].detach().cpu(),
        'fid_score': float(fid_score),
        'last_csv_written': last_csv_written,
        'config': config,
    }
    t.save(checkpoint, EXP_DIR + 'checkpoints/' + 'last_checkpoint.pth')


def compute_instability_index(i):
    if i + 1 < stability_window:
        return t.tensor(float('nan'), device=device)

    start = i + 1 - stability_window
    eps = 1e-8

    fxst_window = f_x_st_record[start:i+1]
    fxq_window = f_x_q_record[start:i+1]

    fxst_cv = fxst_window.std(unbiased=False) / (fxst_window.abs().mean() + eps)
    fxq_cv = fxq_window.std(unbiased=False) / (fxq_window.abs().mean() + eps)

    return 0.5 * (fxst_cv + fxq_cv)

print('Training has started.')
for i in range(start_iter, config['num_train_iters']):
    # obtain positive and negative samples
    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'])

    # calculate ML computational loss d_s_t (Section 3) for data and shortrun samples

    f_x_q_all = f(x_q)
    f_x_q = f_x_q_all.mean()

    f_x_st_all = f(x_s_t)
    f_x_st = f_x_st_all.mean()

    d_s_t = f_x_q - f_x_st

    if config['epsilon'] > 0:
        # scale loss with the langevin implementation
        d_s_t *= 2 / (config['epsilon'] ** 2)
    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record[i] = d_s_t.detach().data
    r_s_t_record[i] = r_s_t
    f_x_q_record[i] = f_x_q.detach()
    f_x_st_record[i] = f_x_st.detach()
    instability_record[i] = compute_instability_index(i)
    
    # compute FID if available
    if feature_extractor is not None and (i == 0 or (i + 1) % config.get('log_freq', 100) == 0):
        # use a bigger batch size for more stable FID estimation if possible
        fid_batch_size = 1000
        x_q_fid = sample_q(batch_size = fid_batch_size)
        x_s_t_fid, _ = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'], update_s_t_0=False, batch_size = fid_batch_size)
        fid_score = compute_fid_batch(x_q_fid, x_s_t_fid, feature_extractor)
        if fid_score > 100 : fid_score = 100 # cap FID at 100 for better visualization of training progress
        fid_record[i] = t.tensor(fid_score).to(device)

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # print and save learning info
    print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}   instability={:>14.9f} fid={:>14.9f}'.format(
        i+1, d_s_t.detach().data, r_s_t, instability_record[i], fid_score))

    if ((i + 1) % config['log_freq'] == 0) or ((i + 1) == config['num_train_iters']):
        csv_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if not csv_exists:
                csv_writer.writerow(['iteration', 'FID', 'DST', 'RST', 'Instability', 'fxq', 'fxst'])
            for j in range(last_csv_written, i + 1):
                csv_writer.writerow([
                    j + 1,
                    float(fid_record[j].detach().cpu()),
                    float(d_s_t_record[j].detach().cpu()),
                    float(r_s_t_record[j].detach().cpu()),
                    float(instability_record[j].detach().cpu()),
                    float(f_x_q_record[j].detach().cpu()),
                    float(f_x_st_record[j].detach().cpu())
                ])
        last_csv_written = i + 1

        # visualize synthesized images
        plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_{:>06d}.png'.format(i+1), x_s_t)
        if config['shortrun_init'] == 'persistent':
            plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']])
        # save network weights
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i + 1))
        # save full training checkpoint for exact resume
        save_full_checkpoint(i)
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, fid_record, f_x_q_record, f_x_st_record,
                             instability_record, EXP_DIR + 'plots/')

    # sample longrun chains to diagnose model steady-state
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(L=config['num_longrun_steps'], init_type=config['longrun_init'], update_s_t_0=False)[0]
        plot_ims(EXP_DIR + 'longrun/' + 'longrun_{:>06d}.png'.format(i+1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i+1))

