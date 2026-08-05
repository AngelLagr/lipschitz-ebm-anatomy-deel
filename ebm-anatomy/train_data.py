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
import torch.nn.functional as F
from deel.torchlip.modules.loss import HKRLoss
from nets import *
from utils import download_flowers_data, plot_ims, plot_diagnostics
from utilfid import *
import torch.fft as fft

# directory for experiment results
EXP_DIR = './out_data/Expe/cifar10_semihalf10Lip_lang10_nofft_lr3e-6/'
CONFIG_FILE = './config_locker/cifar10_Langevin10_semihalfLipDu3FFT.json'
RANDOMRATE = 0.95
K_FACTOR = 2.0 # factor to scale the Lipschitz constant for the K constraint
ALPHA = 0.995 # transition smoothness
FFT_NOISE = False
FFT_NOISE_BANDS = ('low', 'mid', 'high')
HKR_LOSS = False
REGULARIZE = False
REGUL_WEIGHT = 5e-3
HKR_ALPHA = 0.999
LANGEVIN_SQUARED_NORMALISED = False

DST_SCALED = False

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

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
if DST_SCALED:
    temp = 1* (config['epsilon'] ** 2) / 2
else:
    temp = 1
# Initialize FID feature extractor for MNIST
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None


########################
# ## TRAINING SETUP # ##
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'bigtest': BigTestModel, 'semihalf': Semihalf, 'test' : TestModel,'testinv': TestInvModel, 'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet':Wide_ResNet, "wideresnetLip" : Wide_ResNet_LLip, "wideresnetLip_Logsumexp": Wide_ResNet_LLip_Logsumexp, "orthogonium": VanillaNetOrthogonium}
if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip", "orthogonium","test","testinv", "semihalf","bigtest"):
    f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    print(f"Initialized {config['net_type']} with Lip_const={config['Lip_const']}")
elif (config['net_type']=="wideresnet22"):
    f = net_bank["wideresnet"](n_c=config['im_ch'],depth = 22).to(device)
elif (config['net_type']=="wideresnet34"):
    f = net_bank["wideresnet"](n_c=config['im_ch'],depth = 34).to(device)
elif (config['net_type']=="wideresnetLip22"):
    f = net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'],depth = 22).to(device)
else :
    f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)

hkr_loss_fn = HKRLoss(alpha=HKR_ALPHA, min_margin=1.0)
current_lip_const=None

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
WLang_ratio = float(config.get('WLang_ratio', 0.1))
lower_bound_const = float(config.get('lower_bound_const', 1))
upper_bound_const = float(config.get('upper_bound_const', 100))

lip_schedule_enabled = config.get('increase_lip_const', False)
lip_const_start = float(config.get('Lip_const', 10.0))
lip_const_step = float(config.get('Lip_const_step', 0.5))
lip_const_max = float(config.get('Lip_const_max', 500.0))

def update_lip_const(iteration_idx, loss=None, grad=None):
    lip_const = f.Lip_const
    if (not lip_schedule_enabled and not WConstrained_enabled) or not hasattr(f, 'Lip_const'):
        return None

    if lip_schedule_enabled :
        new_lip_const = min(lip_const_max, lip_const_start + lip_const_step * iteration_idx)

    if WConstrained_enabled and not (grad==None) and not (loss==None):        
        #new_lip_const = np.clip(lip_const, (lower_bound_const * WLang_ratio/grad).cpu(), (upper_bound_const * WLang_ratio/grad).cpu())

        buffersize = 60000 if config['data'] == 'mnist' else 50000
        langevin_steps_remaining_in_buffer = config['num_shortrun_steps']/(1-RANDOMRATE) - (config['num_shortrun_steps']/(1-RANDOMRATE)) * (1-(1-(config['batch_size']*(1-RANDOMRATE)/buffersize))**iteration_idx)
        print(f"Estimated Langevin steps remaining in buffer: {langevin_steps_remaining_in_buffer:.2f}")
        print(f"Current Lip_const: {lip_const:.4f}, Loss: {loss:.4f}, Grad: {grad:.4f}")
        new_lip = abs(loss)/(((grad**2)/lip_const) * (max(langevin_steps_remaining_in_buffer, config['num_shortrun_steps']))) 

        new_lip = K_FACTOR * new_lip 
        print(f"Calculated new Lip_const before smoothing: {new_lip:.4f}")
        new_lip_const = ALPHA*lip_const + (1-ALPHA)*new_lip

    f.Lip_const = new_lip_const
    return new_lip_const

# set up optimizer
optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD, 'adamW': t.optim.AdamW}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0 and DST_SCALED:
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

#print size of dataset 
print('Dataset size:', q.shape[0])

def make_freq_mask(sz, band, device):
    y = fft.fftfreq(sz, device=device).reshape(-1, 1)
    x = fft.fftfreq(sz, device=device).reshape(1, -1)
    r2 = x**2 + y**2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04**2))
    elif band == 'mid':
        low = t.exp(-r2 / (2 * 0.04**2))
        high = t.exp(-r2 / (2 * 0.15**2))
        return high - low
    elif band == 'high':
        return 1 - t.exp(-r2 / (2 * 0.15**2))
    raise RuntimeError('Invalid band for frequency mask (use "low", "mid", or "high")')


def sample_freq_noise(batch_size, band='mixed'):
    if band == 'mixed':
        bands = [random.choice(FFT_NOISE_BANDS) for _ in range(batch_size)]
    elif band in FFT_NOISE_BANDS:
        bands = [band] * batch_size
    else:
        raise RuntimeError('Invalid band for frequency noise (use "low", "mid", "high", or "mixed")')

    noise = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device)
    freq = fft.fft2(noise)
    filtered = t.empty_like(noise)

    for current_band in FFT_NOISE_BANDS:
        current_inds = [idx for idx, band_name in enumerate(bands) if band_name == current_band]
        if not current_inds:
            continue
        mask = make_freq_mask(config['im_sz'], current_band, device)[None, None, :, :]
        filtered[current_inds] = fft.ifft2(freq[current_inds] * mask).real

    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
# initialize persistent images from noise (one persistent image for each data image)
# s_t_0 is used when init_type == 'persistent' in sample_s_t()
if FFT_NOISE:
    s_t_0 = sample_freq_noise(q.shape[0], band='mixed')

else :
    s_t_0 = 2 * t.rand_like(q) - 1


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
            # 99% chance each sample comes from s_t_0, 1% chance from uniform random noise
            mask = t.rand(batch_size, device=device) < RANDOMRATE
            indices = t.randperm(s_t_0.shape[0], device=device)[:batch_size]
            persistent_samples = s_t_0[indices]

            if FFT_NOISE:
                random_samples = sample_freq_noise(batch_size, band='mixed')

            else:
                random_samples = 2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device) - 1
            x_s_t_0 = t.where(mask.view(-1, 1, 1, 1), persistent_samples, random_samples)
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
        f_val = f(x_s_t)
        f_prime = t.autograd.grad(f_val.sum(), [x_s_t])[0]
        # / norm^2
        if LANGEVIN_SQUARED_NORMALISED:
            f_prime_norm = f_prime / (f_prime.view(f_prime.shape[0], -1).norm(dim=1).pow(2).view(-1, 1, 1, 1) + 1e-8)
        else :
            f_prime_norm = f_prime

        
        x_s_t.data += - f_prime_norm + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()
        
    if init_type == 'persistent' and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


#Longrun functions
def langevin_grad(f, x_s_t_0_l, config,temp,batch_size):
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    # sampling records
    grads = t.zeros(config['num_longrun_steps'], batch_size)
    ens = t.zeros(config['num_longrun_steps'], batch_size)

    # iterative langevin updates of MCMC samples
    for ell in range(config['num_longrun_steps']):
        f_val = f(x_s_t_l)
        en = f_val / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]

        if LANGEVIN_SQUARED_NORMALISED:
            grad_norm = grad / (grad.view(grad.shape[0], -1).norm(dim=1).pow(2).view(-1, 1, 1, 1) + 1e-8)
        else :
            grad_norm = grad
        
        if DST_SCALED:
            x_s_t_l.data += - ((config['epsilon']**2)/2) * grad_norm + config['epsilon'] * t.randn_like(x_s_t_l)
            grads[ell] = ((config['epsilon']**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        else:
            x_s_t_l.data += - grad_norm + config['epsilon'] * t.randn_like(x_s_t_l)
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()

    return x_s_t_l.detach(), ens, grads

def sample_s_tlongrun(f, config, device, init_type, batch_size):
    if init_type == 'uniform':
        x_s_t_0_l = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0_l = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0_l = sample_freq_noise(batch_size, band='mixed')
    else:
        raise RuntimeError('Invalid method for "init_type" (use "uniform", "gaussian", or "fft")')

    x_s_t_l = x_s_t_0_l.clone()
    
    x_s_t_l = langevin_grad(f, x_s_t_l, config, temp, batch_size)[0]
        

    return x_s_t_l.detach()


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
lip_const_record = t.zeros(config['num_train_iters']).to(device)
lip_const_record[:] = float('nan')
f_x_q_record = t.zeros(config['num_train_iters']).to(device)  # f(x_q).mean()
f_x_st_record = t.zeros(config['num_train_iters']).to(device)  # f(x_s_t).mean()
csv_file_path = EXP_DIR + 'plots/' + 'training_metrics.csv'
last_csv_written = 0
stability_window = 2

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
        'lip_const_record': lip_const_record[:iter_done].detach().cpu(),
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

    logits_q = f(x_q)
    logits_s = f(x_s_t)

    if HKR_LOSS:
        f_x_q = logits_q.mean()
        f_x_st = logits_s.mean()

        kr_term = f_x_q - f_x_st

        hkr_input = t.cat([logits_q, logits_s], dim=0).reshape(-1, 1)
        hkr_target = t.cat([
            -t.ones_like(logits_q),
            t.ones_like(logits_s),
        ], dim=0).reshape(-1, 1)

        loss_raw = hkr_loss_fn(hkr_input, hkr_target)
        loss_saved = loss_raw.detach().clone()
    else:
        f_x_q = logits_q.mean()
        f_x_st = logits_s.mean()
        loss_raw = f_x_q - f_x_st

        if REGULARIZE:
            reg_term =  (logits_q.pow(2).mean() + logits_s.pow(2).mean())
            loss_raw += REGUL_WEIGHT * reg_term

        loss_saved = loss_raw.detach().data.clone()

    # scale loss with the langevin implementation if needed
    if DST_SCALED:
        d_s_t = loss_raw * 2 / (config['epsilon'] ** 2)
    else:
        d_s_t = loss_raw

    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()
    
    #update only after the 200th iteration to allow initial loss and gradient estimates to stabilize
    if (lip_schedule_enabled or WConstrained_enabled) and i >=200:
        current_lip_const = update_lip_const(i, loss=kr_term * r_s_t, grad=r_s_t)

    # record diagnostics
    d_s_t_record[i] = loss_saved
    r_s_t_record[i] = r_s_t
    if (lip_schedule_enabled or WConstrained_enabled):
        lip_const_record[i] = current_lip_const if current_lip_const is not None else float('nan')
    f_x_q_record[i] = f_x_q.detach()
    f_x_st_record[i] = f_x_st.detach()
    print(f"Iteration {i+1}: f(x_q)={f_x_q_record[i]:.4f}, f(x_s_t)={f_x_st_record[i]:.4f}")
    instability_record[i] = compute_instability_index(i)
    
    # compute FID if available
    if feature_extractor is not None and (i == 0 or (i + 1) % config.get('log_freq', 100) == 0):
        if (i==0):
            # set initial FID to a large value 
            fid_record[i] = t.tensor(500).to(device)

        else :
            # use a bigger batch size for more stable FID estimation if possible
            fid_batch_size = 1000
            x_q_fid = sample_q(batch_size = fid_batch_size)
            if FFT_NOISE:
                x_s_t_fid = sample_s_tlongrun(f, config, device, init_type='fft', batch_size = fid_batch_size)
            else:
                x_s_t_fid = sample_s_tlongrun(f, config, device, init_type='uniform', batch_size = fid_batch_size)
            fid_score = compute_fid_batch(x_q_fid, x_s_t_fid, feature_extractor,config['data'] == 'cifar10')
            if fid_score > 500 : fid_score = 500 # cap FID at 100 for better visualization of training progress
            fid_record[i] = t.tensor(fid_score).to(device)

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # print and save learning info
    print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}   instability={:>14.9f} fid={:>14.9f}'.format(
        i+1, d_s_t.detach().data, r_s_t, instability_record[i], fid_score))
    if current_lip_const is not None:
        print('{:>6d}   Lip_const={:>14.9f}'.format(i+1, current_lip_const))

    if ((i + 1) % config['log_freq'] == 0) or ((i + 1) == config['num_train_iters']):
        csv_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if not csv_exists:
                csv_writer.writerow(['iteration', 'FID', 'DST', 'RST', 'Instability', 'fxq', 'fxst', 'Lip_const'])
            for j in range(last_csv_written, i + 1):
                csv_writer.writerow([
                    j + 1,
                    float(fid_record[j].detach().cpu()),
                    float(d_s_t_record[j].detach().cpu()),
                    float(r_s_t_record[j].detach().cpu()),
                    float(instability_record[j].detach().cpu()),
                    float(f_x_q_record[j].detach().cpu()),
                    float(f_x_st_record[j].detach().cpu()),
                    float(lip_const_record[j].detach().cpu())
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

