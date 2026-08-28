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
from ebm_utils import download_flowers_data, plot_ims, plot_diagnostics
from utilfid import *
import torch.fft as fft

# ── experiment paths ────────────────────────────────────────────────────────────
EXP_DIR = './out_data/Expe/cifar10_lang10_dim4x8x16x32/'
CONFIG_FILE = './config_locker/cifar10_Langevin10coarse.json'

# ── sampling hyper-parameters ───────────────────────────────────────────────────
RANDOMRATE = 0.95
LANGEVIN_SQUARED_NORMALISED = False

# ─ Lipschitz parameters ─────────────────────────────────────────────────────────
K_FACTOR = 2.0 # factor to scale the Lipschitz constant for the K constraint
# To activate the automatic update of the Lipschitz constraint (which does not scale during our tests)
# you need to add in the config file : Wasserstein_LipConstrained = True

# If you want to activate the manual Lip scheduling add in the config file: 
# - increase_lip_const = True
# - Lip_const_step = a float
# - Lip_const_max = a float
ALPHA = 0.995 # transition smoothness of the updating

# ─ use FFT-filtered noise or not ────────────────────────────────────────────────
FFT_NOISE = False
FFT_NOISE_BANDS = ('low', 'mid', 'high')

# ─ Loss parameters ──────────────────────────────────────────────────────────────
HKR_LOSS = False
REGULARIZE = True
REGUL_WEIGHT = 1e-2
HKR_ALPHA = 0.999
DST_SCALED = False # doesn't change anything, it was True on the original code from Nijkamp.

# ─ MULTI-RESOLUTION CASCADE ─────────────────────────────────────────────────────
# DIM_RESIZE=True   -> multi-résolution cascade. We just give the list of resolutions to traverse 
#                      (increasing, the last value MUST be equal to config['im_sz']) and the number
#                      of Langevin steps for each level BEYOND the first (the first level always uses
#                      config['num_shortrun_steps'], like the original "noise -> data" model).
# Examples :
#   4 steps : noise -> 2 -> 4 -> 8 -> 16 -> 32
#       CASCADE_RESOLUTIONS = [2, 4, 8, 16, 32]
#       LEVEL_LRS = [9e-4, 6e-4, 4e-4, 3e-4, 2e-4]
#       CASCADE_STEPS       = [10, 10, 10, 5]  # 2->4, 4->8, 8->16, 16->32 # the number of steps for the first level (noise -> 2) is always config['num_shortrun_steps'] and is not included in CASCADE_STEPS.
#
# len(CASCADE_STEPS) = len(CASCADE_RESOLUTIONS) - 1.
DIM_RESIZE = True
CASCADE_RESOLUTIONS = [2, 4, 8, 16, 32]
# we obtained during our test that the lower the details resolutions are, the more steps are needed to learn the details.
CASCADE_STEPS = [20, 15, 10, 10]
# it appear that if the lr is too low in early levels, the model will not learn anything and will diverge
LEVEL_LRS = [9e-4, 6e-4, 4e-4, 3e-4, 2e-4]
# With 5 WideResnet (700k param each) we achieve 3.5million parameters

#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

if not DIM_RESIZE:
    # mode "un seul niveau" : pas de cascade, on entraîne directement à la résolution finale
    CASCADE_RESOLUTIONS = [config['im_sz']]
    CASCADE_STEPS = []

# validation de la config de la cascade
assert CASCADE_RESOLUTIONS[-1] == config['im_sz'], \
    'La dernière résolution de CASCADE_RESOLUTIONS doit être égale à config["im_sz"] ({}).'.format(config['im_sz'])
assert len(CASCADE_STEPS) == len(CASCADE_RESOLUTIONS) - 1, \
    'CASCADE_STEPS doit contenir exactement len(CASCADE_RESOLUTIONS)-1 valeurs (une par transition).'
for a, b in zip(CASCADE_RESOLUTIONS[:-1], CASCADE_RESOLUTIONS[1:]):
    assert a < b, 'CASCADE_RESOLUTIONS doit être strictement croissante ({} >= {}).'.format(a, b)
N_LEVELS = len(CASCADE_RESOLUTIONS)

assert len(LEVEL_LRS) == N_LEVELS, \
    'LEVEL_LRS must contain exactly N_LEVELS ({}) values, one per cascade level.'.format(N_LEVELS)


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
    for file in ['train_data_coarsetofine.py', 'nets.py', 'ebm_utils.py', CONFIG_FILE]:
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
net_bank = {'bigtest': BigTestModel, 'semihalf': Semihalf, 'test' : TestModel, 'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet':Wide_ResNet, "wideresnetLip" : Wide_ResNet_LLip, "wideresnetLip_Logsumexp": Wide_ResNet_LLip_Logsumexp, "orthogonium": VanillaNetOrthogonium}

def build_net():
    """Construit un réseau selon config['net_type'], avec la même logique de sélection
    que le modèle de base (factorisée pour pouvoir instancier autant de niveaux que voulu)."""
    if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip", "orthogonium", "test", "testinv", "semihalf", "bigtest"):
        net = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    elif (config['net_type'] == "wideresnet22"):
        net = net_bank["wideresnet"](n_c=config['im_ch'], depth=22).to(device)
    elif (config['net_type'] == "wideresnet34"):
        net = net_bank["wideresnet"](n_c=config['im_ch'], depth=34).to(device)
    elif (config['net_type'] == "wideresnetLip22"):
        net = net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'], depth=22).to(device)
    elif (config['net_type'] == "wideresnetsmaller"):
        net = net_bank["wideresnet"](n_c=config['im_ch'], depth=10, widen_factor=2).to(device)
    else:
        net = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
    return net

# ── un réseau + un optimiseur par niveau de la cascade ─────────────────────────
nets = [build_net() for _ in range(N_LEVELS)]

optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD, 'adamW': t.optim.AdamW}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0 and DST_SCALED:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2

# let the user choose the first learning rate of the first model with FIRST_EBM_LR, all the other hyperparameters are set in the config file

optims = [optim_bank[config['optimizer_type']](net.parameters(), lr=LEVEL_LRS[idx])
          for idx, net in enumerate(nets)]
# alias pour conserver la compatibilité avec tout le code "niveau 0" ci-dessous
# (HKR loss, scheduling du Lip const, échantillonnage longrun, FID, ...)
f = nets[0]
optim = optims[0]

hkr_loss_fn = HKRLoss(alpha=HKR_ALPHA, min_margin=1.0)
current_lip_const=None

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)

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


print('Processing data...')
# make tensor of training data
if config['data'] == 'flowers':
    download_flowers_data()
data = {'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func)}


def build_level_transform(res, is_final):
    """Pipeline de transform pour un niveau de résolution donné.
    - niveaux intermédiaires : downsample à `res` puis ré-agrandissement en blocs
      (nearest) vers config['im_sz'].
    - dernier niveau : image à sa vraie résolution finale, sans effet "blocky"."""
    if is_final:
        resize_steps = [tr.Resize(config['im_sz'])]
    else:
        resize_steps = [
            tr.Resize(res),
            tr.Resize(config['im_sz'], interpolation=tr.InterpolationMode.NEAREST),
        ]
    return tr.Compose(resize_steps + [
        tr.CenterCrop(config['im_sz']),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), tuple(0.5*t.ones(config['im_ch'])))])


# ── un dataset (positifs) par niveau de la cascade ──────────────────────────────
level_data = []
for idx, res in enumerate(CASCADE_RESOLUTIONS):
    is_final = (idx == N_LEVELS - 1)
    transform_res = build_level_transform(res, is_final)
    ds = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform_res)]).to(device)
    level_data.append(ds)
    tag = 'résolution réelle' if is_final else 'blocky {0}x{0} -> {1}x{1}'.format(res, config['im_sz'])
    print('Dataset niveau {} (détail {}) [{}] : {} images'.format(idx, res, tag, ds.shape[0]))

# alias pour compatibilité avec le code "niveau 0" (sample_q, s_t_0, ...)
q = level_data[0]


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
# s_t_0 is used when init_type == 'persistent' in sample_s_t() -- uniquement pour le niveau 0
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
# (utilisé uniquement pour le niveau 0 de la cascade : c'est le seul qui part du bruit /
#  d'un buffer persistant, exactement comme le modèle "noise -> data" d'origine)
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


# ── Langevin générique pour les niveaux >= 1 de la cascade ──────────────────────
# Init SANS bruit, directement depuis un batch réel du dataset "source" (résolution
# du niveau précédent), puis L pas de Langevin classiques (bruit epsilon à chaque
# pas, comme dans sample_s_t). Réutilisé pour chaque transition de la cascade.
def sample_s_t_from_source(net, source_data, L, batch_size=config['batch_size']):
    # init déterministe : un batch réel pris tel quel dans source_data (pas de bruit ajouté)
    x_s_t_0, _ = sample_image_set(source_data, batch_size=batch_size)

    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).to(device)
    for ell in range(L):
        f_val = net(x_s_t)
        f_prime = t.autograd.grad(f_val.sum(), [x_s_t])[0]
        if LANGEVIN_SQUARED_NORMALISED:
            f_prime_norm = f_prime / (f_prime.view(f_prime.shape[0], -1).norm(dim=1).pow(2).view(-1, 1, 1, 1) + 1e-8)
        else:
            f_prime_norm = f_prime

        x_s_t.data += - f_prime_norm + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

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

# containers for diagnostic records (see Section 3) -- niveau 0 (comportement d'origine)
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

# ── records diagnostiques génériques pour les niveaux >= 1 de la cascade ────────
# un tenseur par niveau, indexé par le numéro du niveau (1 .. N_LEVELS-1)
cascade_d_s_t_record = {lvl: t.zeros(config['num_train_iters']).to(device) for lvl in range(1, N_LEVELS)}
cascade_r_s_t_record = {lvl: t.zeros(config['num_train_iters']).to(device) for lvl in range(1, N_LEVELS)}


def save_full_checkpoint(iteration_idx):
    iter_done = iteration_idx + 1
    checkpoint = {
        'iter': iter_done,
        'cascade_resolutions': CASCADE_RESOLUTIONS,
        'cascade_steps': CASCADE_STEPS,
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
    # ── ajout générique des niveaux >= 1 de la cascade ──
    for lvl in range(1, N_LEVELS):
        res = CASCADE_RESOLUTIONS[lvl]
        checkpoint['model_res{}_state_dict'.format(res)] = nets[lvl].state_dict()
        checkpoint['optimizer_res{}_state_dict'.format(res)] = optims[lvl].state_dict()
        checkpoint['d_s_t_res{}_record'.format(res)] = cascade_d_s_t_record[lvl][:iter_done].detach().cpu()
        checkpoint['r_s_t_res{}_record'.format(res)] = cascade_r_s_t_record[lvl][:iter_done].detach().cpu()

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

print('Training has started. Cascade : {}'.format(' -> '.join(str(r) for r in CASCADE_RESOLUTIONS)))
for i in range(start_iter, config['num_train_iters']):

    # ═══════════════════════════════════════════════════════════════════════
    # ── NIVEAU 0 : bruit -> première résolution (comportement d'origine) ──
    # ═══════════════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════════════
    # ── NIVEAUX >= 1 de la cascade : boucle générique sur chaque transition ─
    # (résolution[lvl-1] -> résolution[lvl], entraînement simple type contraste
    #  d_s_t = f(positif) - f(négatif), sans HKR/regularize, comme f16/f32
    #  dans le code d'origine) ─
    # ═══════════════════════════════════════════════════════════════════════
    x_neg_per_level = {}  # pour la visualisation plus bas
    for lvl in range(1, N_LEVELS):
        net_lvl = nets[lvl]
        optim_lvl = optims[lvl]
        res_prev = CASCADE_RESOLUTIONS[lvl - 1]
        res_lvl = CASCADE_RESOLUTIONS[lvl]
        steps_lvl = CASCADE_STEPS[lvl - 1]
        source_data = level_data[lvl - 1]   # dataset du niveau précédent (pour init des négatifs)
        positive_data = level_data[lvl]     # dataset du niveau courant (positifs)

        # positifs : dataset du niveau courant (+ bruit comme sample_q)
        x_q_lvl = sample_image_set(positive_data)[0] + config['data_epsilon'] * t.randn_like(x_q)
        # négatifs : init sans bruit depuis le dataset du niveau précédent, puis Langevin
        x_neg_lvl, r_s_t_lvl = sample_s_t_from_source(net_lvl, source_data, L=steps_lvl)

        f_x_q_lvl = net_lvl(x_q_lvl).mean()
        f_x_neg_lvl = net_lvl(x_neg_lvl).mean()
        d_s_t_lvl = f_x_q_lvl - f_x_neg_lvl

        optim_lvl.zero_grad()
        d_s_t_lvl.backward()
        optim_lvl.step()

        cascade_d_s_t_record[lvl][i] = d_s_t_lvl.detach()
        cascade_r_s_t_record[lvl][i] = r_s_t_lvl
        x_neg_per_level[lvl] = x_neg_lvl

        print('{:>6d}   [{}->{}]  d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(
            i+1, res_prev, res_lvl, d_s_t_lvl.detach().data, r_s_t_lvl))

        # anneal learning rate (même politique que le niveau 0)
        for lr_gp in optim_lvl.param_groups:
            lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # compute FID if available (uniquement pour le niveau 0, comme dans le code d'origine)
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

    # anneal learning rate (niveau 0)
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
                header = ['iteration', 'FID', 'DST', 'RST', 'Instability', 'fxq', 'fxst', 'Lip_const']
                for lvl in range(1, N_LEVELS):
                    res = CASCADE_RESOLUTIONS[lvl]
                    header += ['DST_res{}'.format(res), 'RST_res{}'.format(res)]
                csv_writer.writerow(header)
            for j in range(last_csv_written, i + 1):
                row = [
                    j + 1,
                    float(fid_record[j].detach().cpu()),
                    float(d_s_t_record[j].detach().cpu()),
                    float(r_s_t_record[j].detach().cpu()),
                    float(instability_record[j].detach().cpu()),
                    float(f_x_q_record[j].detach().cpu()),
                    float(f_x_st_record[j].detach().cpu()),
                    float(lip_const_record[j].detach().cpu()),
                ]
                for lvl in range(1, N_LEVELS):
                    row += [
                        float(cascade_d_s_t_record[lvl][j].detach().cpu()),
                        float(cascade_r_s_t_record[lvl][j].detach().cpu()),
                    ]
                csv_writer.writerow(row)
        last_csv_written = i + 1

        # visualize synthesized images (niveau 0)
        plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_{:>06d}.png'.format(i+1), x_s_t)
        if config['shortrun_init'] == 'persistent':
            plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']])

        # visualisation générique des négatifs pour chaque niveau >= 1
        for lvl in range(1, N_LEVELS):
            res_prev = CASCADE_RESOLUTIONS[lvl - 1]
            res_lvl = CASCADE_RESOLUTIONS[lvl]
            plot_ims(EXP_DIR + 'shortrun/' + 'x_neg_{}to{}_{:>06d}.png'.format(res_prev, res_lvl, i+1),
                     x_neg_per_level[lvl])

        # save network weights (niveau 0)
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i + 1))
        # sauvegarde générique des poids pour chaque niveau >= 1
        for lvl in range(1, N_LEVELS):
            res = CASCADE_RESOLUTIONS[lvl]
            t.save(nets[lvl].state_dict(),
                   EXP_DIR + 'checkpoints/' + 'net_res{}_{:>06d}.pth'.format(res, i + 1))

        # save full training checkpoint for exact resume
        save_full_checkpoint(i)
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t (niveau 0)
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, fid_record, f_x_q_record, f_x_st_record,
                             instability_record, EXP_DIR + 'plots/')

    # sample longrun chains to diagnose model steady-state (niveau 0)
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(L=config['num_longrun_steps'], init_type=config['longrun_init'], update_s_t_0=False)[0]
        plot_ims(EXP_DIR + 'longrun/' + 'longrun_{:>06d}.png'.format(i+1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i+1))