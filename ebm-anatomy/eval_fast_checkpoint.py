import json
import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch.fft as fft

from utils import*
from nets import *
from utilfid import *

# =========================
# PARAMETRES (EN HAUT)
# =========================
# parse the args and set up experiment directory


EXP_DIR = './out_data/Expe/cifar10_lip_lang10_HKR/'

# find any json config file in the code directory of the experiment
CONFIG_FILE = None
for file in os.listdir(EXP_DIR+'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR+'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}")

CHECKPOINT_FILE = EXP_DIR + 'checkpoints/last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 3000
LONGRUN_STEPS = 400
LONGRUN_INIT = 'fft' 
FFT_NOISE_BANDS = ('low', 'mid', 'high')
SEED = 42
# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
lip_const = None
# temperature from training
if config['epsilon'] > 0:
    temp = 1* (config['epsilon'] ** 2) / 2
else:
    temp = 1


############################################
# ## INIT PERSISTENT FROM MNIST (always) ## #
############################################
def langevin_grad(f, x_s_t_0_l, config, temp_l, lip_const=None):
    if WConstrained_enabled:
        f.Lip_const = lip_const
        
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    actual_batch_size = x_s_t_0_l.shape[0]  
    grads = t.zeros(LONGRUN_STEPS, actual_batch_size)
    ens = t.zeros(LONGRUN_STEPS, actual_batch_size)

    # iterative langevin updates of MCMC samples
    for ell in range(LONGRUN_STEPS):
        f_val = f(x_s_t_l)
        en = f_val / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += - ((config['epsilon']**2)/2) * grad + config['epsilon'] * t.randn_like(x_s_t_l)
        grads[ell] = ((config['epsilon']**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        
        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')
    return x_s_t_l.detach(), ens, grads



def sample_q(q, data_epsilon, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + data_epsilon * t.randn_like(x_q)


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


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]

def sample_s_t(f, config, device, L, init_type, batch_size, lip_const = None):
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0 = sample_freq_noise(batch_size, band='mixed')
    else:   
        raise RuntimeError('Invalid init_type for sample_s_t (use "uniform", "gaussian", or "fft")')

    results = []
    grad_results = []
    for i in range(0, batch_size, 1000):
        print(f"\n batch chunk :{i} \n")
        chunk = x_s_t_0[i:i+1000]
        out, _, grads = langevin_grad(f, chunk, config, temp, lip_const)
        results.append(out.cpu())
        grad_results.append(grads)
        t.cuda.empty_cache()

    return t.cat(results, dim=0).to(device), t.cat(grad_results, dim=1)



# -------- build model --------
net_bank = {'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet':Wide_ResNet,'wideresnetSoftplus': Wide_ResNetSoftPlus, "wideresnetLip" : Wide_ResNet_LLip, "wideresnetLipSoftplus": Wide_ResNet_LLipSoftPlus, "orthogonium": VanillaNetOrthogonium}
if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip", "wideresnetLipSoftplus"):
    f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    current_lip_const=None
elif (config['net_type']=="wideresnet22"):
    f = net_bank["wideresnet"](n_c=config['im_ch'],depth = 22).to(device)
    current_lip_const=None
elif (config['net_type']=="wideresnet34"):
    f = net_bank["wideresnet"](n_c=config['im_ch'],depth = 34).to(device)
    current_lip_const=None
elif (config['net_type']=="wideresnetLip22"):
    f = net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'],depth = 22).to(device)
else :
    f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
    current_lip_const=None
# -------- load checkpoint --------
checkpoint = t.load(CHECKPOINT_FILE, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    f.load_state_dict(checkpoint['model_state_dict'])
    if WConstrained_enabled:
        lip_const = float(checkpoint['lip_const_record'][-1])
        print("Const lip =", lip_const)
else:
    f.load_state_dict(checkpoint)
    if WConstrained_enabled:
        lip_const = float(input("Give the last Lip constant value during training : "))
        print("Const lip =", lip_const)



f.eval()
f.to(device)
t.cuda.empty_cache()  

seed = SEED
t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
import random
import numpy as np
np.random.seed(SEED)
random.seed(SEED)

# -------- load data --------
data = {
    'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
    'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
    'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
}
transform = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
])
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

# -------- compute FID batch=10000 --------
# Initialize FID feature extractor for MNIST
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None

with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)



# -------- compute long-run FID from noise init --------
x_s_t_longrun = sample_s_t(
    f=f,
    config=config,
    device=device,
    L=LONGRUN_STEPS,
    init_type=LONGRUN_INIT,
    batch_size=FID_BATCH_SIZE,
    lip_const = lip_const
)

x_s_t_longrun, longrun_grads = x_s_t_longrun
#scale
longrun_grads = longrun_grads.detach()

plt.figure(figsize=(8, 4))
plt.plot(longrun_grads.mean(dim=1).cpu())
plt.xlabel('Langevin step')
plt.ylabel('Mean gradient norm')
plt.tight_layout()
plt.savefig(EXP_DIR + 'longrun/' + 'langevin_grads.png', dpi=150)
plt.close()

#save a litle part of the long-run image for visualisation
x_s_t_longrun_vis = x_s_t_longrun[:16]
plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample.png', x_s_t_longrun_vis)
plot_ims(EXP_DIR + 'longrun/' + 'q_sample.png', x_q_fid[:16])

fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor,config['data'] == 'cifar10')

#compute the is score
is_score = compute_is_batch(x_q_fid, x_s_t_longrun, feature_extractor, config['data'] == 'cifar10')

print(EXP_DIR)
print(CHECKPOINT_FILE)
print(f'FID_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {fid_longrun:.6f}')
print(f'IS_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {is_score:.6f}')

# -------- read csv and compute slope a --------
df = pd.read_csv(CSV_FILE)

iters = df['iteration'].to_numpy()
fids = df['FID'].to_numpy()

#Calculate the 3 best FID without taking into account the NaN values
fids_no_nan = fids[~pd.isna(fids)]
iters_no_nan = iters[~pd.isna(fids)]
top_k = min(3, len(fids_no_nan))
best_idx = np.argsort(fids_no_nan)[:top_k]
best_fids = fids_no_nan[best_idx]
best_iters = iters_no_nan[best_idx]
best_fid = best_fids[0]
best_iter = best_iters[0]

#print the 3 best FID points on the curve
print(f'Top {top_k} FID points:')
for i in range(top_k):
    print(f'Iter={best_iters[i]:.0f}, FID={best_fids[i]:.6f}')

# take the first point and the point where FID <= 10
idxfirst = 0
idx10 = next(i for i, v in enumerate(fids) if v <= 100)

x1, y1 = float(iters[idxfirst]), float(fids[idxfirst])
x2, y2 = float(iters[idx10]), float(fids[idx10])

a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

print(f'First point: iter={x1:.0f}, fid={y1:.6f}')
print(f'Point FID<=10: iter={x2:.0f}, fid={y2:.6f}')
print(f'Droite: f(x) = a*x + b avec a={a:.8f}, b={b:.8f}')

