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

EXP_DIR = './out_data/Expe/cifar10_50&20WARMUP_TESTCLip_Langevin10_Seed42NOISEFFT_95_WIDEMODEL/'
CONFIG_FILE = None
for file in os.listdir(EXP_DIR+'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR+'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}")

CHECKPOINT_FILE = EXP_DIR + 'checkpoints/net_196000.pth'
LIP_MODEL_FILE = EXP_DIR + 'checkpoints/lip_model.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 3000
LONGRUN_STEPS = 400
LONGRUN_INIT = 'fft'
PHASE2_NET_TYPE = 'wide3resnet22'
WARMUP_LANGEVIN_STEPS = 50
FFT_NOISE_BANDS = ('low', 'mid', 'high')
SEED = 42
LIP_CONST = 0.97

# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if config['epsilon'] > 0:
    temp = 1 * (config['epsilon'] ** 2) / 2
else:
    temp = 1


############################################
# ## LANGEVIN DYNAMICS #################### #
############################################
def langevin_grad(f, x_s_t_0_l, config, temp_l):
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    actual_batch_size = x_s_t_0_l.shape[0]  
    grads = t.zeros(LONGRUN_STEPS, actual_batch_size)
    ens = t.zeros(LONGRUN_STEPS, actual_batch_size)

    for ell in range(LONGRUN_STEPS):
        en = f(x_s_t_l) / temp_l
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += - ((config['epsilon']**2)/2) * grad + config['epsilon'] * t.randn_like(x_s_t_l)
        grads[ell] = ((config['epsilon']**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        
        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')
    return x_s_t_l.detach(), ens, grads


def langevin_refine_with_model(model, x_init, num_steps):
    """Refine samples using a given model for num_steps Langevin iterations."""
    if num_steps <= 0:
        return x_init.detach()

    x = t.autograd.Variable(x_init.clone().detach(), requires_grad=True)
    for _ in range(num_steps):
        grad = t.autograd.grad(model(x).sum(), [x])[0]
        x = (x - grad + config['epsilon'] * t.randn_like(x)).detach().requires_grad_(True)
    return x.detach()


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


def sample_q(q, data_epsilon, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + data_epsilon * t.randn_like(x_q)


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]


def sample_s_t_with_warmup(f_main, f_lip, config, device, L, init_type, batch_size):
    """Sample using two-stage: warmup with Lip model, then refine with main model."""
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0 = sample_freq_noise(batch_size, band='mixed')
    else:   
        raise RuntimeError('Invalid init_type (use "uniform", "gaussian", or "fft")')

    # Stage 1: Warmup with Lipschitz model
    if f_lip is not None and WARMUP_LANGEVIN_STEPS > 0:
        print(f"\nWarmup stage: refining {batch_size} samples with Lip model ({WARMUP_LANGEVIN_STEPS} steps)...")
        warmup_results = []
        for i in range(0, batch_size, 1000):
            print(f"Warmup chunk: {i}")
            chunk = x_s_t_0[i:i+1000]
            refined_chunk = langevin_refine_with_model(f_lip, chunk, WARMUP_LANGEVIN_STEPS)
            warmup_results.append(refined_chunk.cpu())
            t.cuda.empty_cache()
        x_s_t_0 = t.cat(warmup_results, dim=0).to(device)

    #save warmup samples for visualization
    x_s_t_0_vis = x_s_t_0[:16]
    plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample_warmup_init.png', x_s_t_0_vis)
    # Stage 2: Main refinement with phase 2 model
    results = []
    grad_results = []
    for i in range(0, batch_size, 1000):
        print(f"Main refinement chunk: {i}")
        chunk = x_s_t_0[i:i+1000]
        out, _, grads = langevin_grad(f_main, chunk, config, temp)
        results.append(out.cpu())
        grad_results.append(grads)
        t.cuda.empty_cache()

    return t.cat(results, dim=0).to(device), t.cat(grad_results, dim=1)


# -------- build and load models --------
net_bank = {'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,
            'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 
            'wideresnet':Wide_ResNet, "wideresnetLip" : Wide_ResNet_LLip, "orthogonium": VanillaNetOrthogonium}

def build_model(net_type):
    if net_type in ("vanillaLip", "vanillaLip2", "wideresnetLip", "orthogonium"):
        return net_bank[net_type](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    if net_type == "wideresnet22":
        return net_bank["wideresnet"](n_c=config['im_ch'], depth=22).to(device)
    if net_type == "wide3resnet22":
        return net_bank["wideresnet"](n_c=config['im_ch'], depth=22, widen_factor=3).to(device)
    if net_type == "wideresnet34":
        return net_bank["wideresnet"](n_c=config['im_ch'], depth=34).to(device)
    if net_type == "wideresnetLip22":
        return net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'], depth=22).to(device)
    if net_type == "wide3resnetLip":
        return net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'], depth=16, widen_factor=3).to(device)
    return net_bank[net_type](n_c=config['im_ch']).to(device)

# Load main model (phase 2)
checkpoint = t.load(CHECKPOINT_FILE, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    f_main = build_model(PHASE2_NET_TYPE)
    f_main.load_state_dict(checkpoint['model_state_dict'])
else:
    f_main = build_model(PHASE2_NET_TYPE)
    f_main.load_state_dict(checkpoint)

# Load Lip model (phase 1)
f_lip = None
if os.path.exists(LIP_MODEL_FILE):
    f_lip = build_model(config['net_type'])
    f_lip.load_state_dict(t.load(LIP_MODEL_FILE, map_location=device))
    f_lip.eval()
    for param in f_lip.parameters():
        param.requires_grad_(False)
    print(f"Loaded Lip model from {LIP_MODEL_FILE}")
else:
    print(f"Warning: Lip model not found at {LIP_MODEL_FILE}. Proceeding without warmup.")

f_main.eval()
f_main.to(device)
t.cuda.empty_cache()

# Set seed
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

# -------- Initialize FID feature extractor --------
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None

# -------- Sample from data distribution --------
with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)

# -------- Generate samples using warmup pipeline --------
print(f"\nGenerating {FID_BATCH_SIZE} samples with warmup pipeline...")
x_s_t_longrun, longrun_grads = sample_s_t_with_warmup(
    f_main=f_main,
    f_lip=f_lip,
    config=config,
    device=device,
    L=LONGRUN_STEPS,
    init_type=LONGRUN_INIT,
    batch_size=FID_BATCH_SIZE
)

# Plot gradient evolution
plt.figure(figsize=(8, 4))
plt.plot(longrun_grads.mean(dim=1).cpu())
plt.xlabel('Langevin step')
plt.ylabel('Mean gradient norm')
plt.title('Gradient Evolution (Warmup Pipeline)')
plt.tight_layout()
plt.savefig(EXP_DIR + 'longrun/' + 'langevin_grads_warmup.png', dpi=150)
plt.close()

# Save sample visualizations
x_s_t_longrun_vis = x_s_t_longrun[:16]
plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample_warmup.png', x_s_t_longrun_vis)
plot_ims(EXP_DIR + 'longrun/' + 'q_sample_warmup.png', x_q_fid[:16])

# Compute FID and IS scores
fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor, config['data'] == 'cifar10')
is_score = compute_is_batch(x_q_fid, x_s_t_longrun, feature_extractor, config['data'] == 'cifar10')

print(EXP_DIR)
print(f'FID_longrun_warmup(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, warmup={WARMUP_LANGEVIN_STEPS}) = {fid_longrun:.6f}')
print(f'IS_longrun_warmup(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, warmup={WARMUP_LANGEVIN_STEPS}) = {is_score:.6f}')
