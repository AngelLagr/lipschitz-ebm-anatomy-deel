import json
import os
import random

import matplotlib.pyplot as plt
import torch as t
import torch.fft as fft
import torchvision.datasets as datasets
import torchvision.transforms as tr
from utils import*
from nets import *


EXP_DIR = './out_data/Expe/cifar10_lip_lang10_HKR_REAL_alpha09/'
BATCH_SIZE = 1000
REFINE_STEPS = 400
SEED = 42
FFT_NOISE_BANDS = ('low', 'mid', 'high')


device = t.device('cuda' if t.cuda.is_available() else 'cpu')
t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
random.seed(SEED)


def find_config(exp_dir):
    code_dir = os.path.join(exp_dir, 'code')
    for name in os.listdir(code_dir):
        if name.endswith('.json'):
            return os.path.join(code_dir, name)
    raise RuntimeError(f'No JSON config found in {code_dir}')


def find_checkpoint(exp_dir):
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    last_ckpt = os.path.join(checkpoints_dir, 'last_checkpoint.pth')
    if os.path.exists(last_ckpt):
        return last_ckpt
    cands = sorted([n for n in os.listdir(checkpoints_dir) if n.startswith('net_') and n.endswith('.pth')])
    if not cands:
        raise RuntimeError(f'No checkpoint found in {checkpoints_dir}')
    return os.path.join(checkpoints_dir, cands[-1])


def build_model(config):
    net_bank = {
        'vanilla': VanillaNet,
        'vanillaLip': VanillaNetLLip,
        'nonlocal': NonlocalNet,
        'vanillaHalfLip': VanillaNetHalfLip,
        'vanillaLip2': VanillaNetLLip2,
        'vanilla2': VanillaNet2,
        'wideresnet': Wide_ResNet,
        'wideresnetSoftplus': Wide_ResNetSoftPlus,
        'wideresnetLip': Wide_ResNet_LLip,
        'wideresnetLipSoftplus': Wide_ResNet_LLipSoftPlus,
        'orthogonium': VanillaNetOrthogonium,
    }

    net_type = config['net_type']
    if net_type in ('vanillaLip', 'vanillaLip2', 'wideresnetLip', 'orthogonium'):
        return net_bank[net_type](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    if net_type == 'wideresnet22':
        return net_bank['wideresnet'](n_c=config['im_ch'], depth=22).to(device)
    if net_type == 'wideresnet34':
        return net_bank['wideresnet'](n_c=config['im_ch'], depth=34).to(device)
    if net_type == 'wideresnetLip22':
        return net_bank['wideresnetLip'](n_c=config['im_ch'], Lip_const=config['Lip_const'], depth=22).to(device)
    return net_bank[net_type](n_c=config['im_ch']).to(device)


def load_model(config, checkpoint_path):
    model = build_model(config)
    ckpt = t.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def make_freq_mask(sz, band):
    y = fft.fftfreq(sz, device=device).reshape(-1, 1)
    x = fft.fftfreq(sz, device=device).reshape(1, -1)
    r2 = x ** 2 + y ** 2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04 ** 2))
    if band == 'mid':
        low = t.exp(-r2 / (2 * 0.04 ** 2))
        high = t.exp(-r2 / (2 * 0.15 ** 2))
        return high - low
    if band == 'high':
        return 1 - t.exp(-r2 / (2 * 0.15 ** 2))
    raise RuntimeError('Invalid FFT band')


def sample_freq_noise(config, batch_size):
    bands = [random.choice(FFT_NOISE_BANDS) for _ in range(batch_size)]
    noise = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device)
    freq = fft.fft2(noise)
    filtered = t.empty_like(noise)

    for band in FFT_NOISE_BANDS:
        inds = [i for i, b in enumerate(bands) if b == band]
        if not inds:
            continue
        mask = make_freq_mask(config['im_sz'], band)[None, None, :, :]
        filtered[inds] = fft.ifft2(freq[inds] * mask).real

    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


def sample_data_batch(config, batch_size):
    transform = tr.Compose([
        tr.Resize(config['im_sz']),
        tr.CenterCrop(config['im_sz']),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
    ])
    ds = datasets.CIFAR10(root='./data/cifar10', transform=transform, download=True)
    inds = t.randperm(len(ds))[:batch_size].tolist()
    return t.stack([ds[i][0] for i in inds]).to(device)


def langevin_refine(model, x0, eps, steps):
    x = x0.clone().detach().requires_grad_(True)
    for _ in range(steps):
        energy = model(x)
        grad = t.autograd.grad(energy.sum(), [x])[0]
        x = (x - grad + eps * t.randn_like(x)).detach().requires_grad_(True)
    return x.detach()


def main():
    config_path = find_config(EXP_DIR)
    ckpt_path = find_checkpoint(EXP_DIR)

    with open(config_path) as f:
        config = json.load(f)

    model = load_model(config, ckpt_path)
    x_data = sample_data_batch(config, BATCH_SIZE)
    x_noise = sample_freq_noise(config, BATCH_SIZE)
    x_refined = langevin_refine(model, x_noise, config['epsilon'], REFINE_STEPS)

    with t.no_grad():
        hkr_data = model(x_data).view(-1).cpu().numpy()
        hkr_noise = model(x_noise).view(-1).cpu().numpy()
        hkr_refined = model(x_refined).view(-1).cpu().numpy()

    out_dir = os.path.join(EXP_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'hkr_hist_simple.png')

    plt.figure(figsize=(8, 4))
    plt.hist(hkr_data, bins=50, alpha=0.5, label='data')
    plt.hist(hkr_noise, bins=50, alpha=0.5, label='noise')
    plt.hist(hkr_refined, bins=50, alpha=0.5, label='noise refined shortrun')
    plt.legend()
    plt.title('HKR values histogram (data / noise / noise refined)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

    print('Checkpoint:', ckpt_path)
    print('Saved:', out_png)

    # Try to do a special refinement : xst=xst - (E_lip(x) + epslion) * grad(E_lip)/(norm(gradElip)^2) for 1 step and plot it
    x_special = x_noise.clone().detach().requires_grad_(True)
    energy_lip = model(x_special)
    grad_lip = t.autograd.grad(energy_lip.sum(), [x_special])[0]
    print('Energy Lip:', energy_lip.mean().item())
    step_size = (energy_lip) / (grad_lip.view(grad_lip.size(0), -1).norm(dim=1) ** 2 + 1e-8)
    x_special_refined = (x_special - step_size.view(-1, 1, 1, 1) * grad_lip).detach()
    
    #save the sample refined in a png
    out_png_special = os.path.join(out_dir, 'hkr_specialfromnoise.png')
    x_special_refined = x_special_refined[:16]
    plot_ims(out_png_special, x_special_refined)
    
    # do it again but with a loop for 10 steps
    x_special_refined_loop = x_special.clone().detach().requires_grad_(True)


    for _ in range(10):
        energy_lip_loop = model(x_special_refined_loop)
        print('Energy Lip loop:', energy_lip_loop.mean().item())
        grad_lip_loop = t.autograd.grad(energy_lip_loop.sum(), [x_special_refined_loop])[0]
        step_size_loop = (energy_lip_loop) / (grad_lip_loop.view(grad_lip_loop.size(0), -1).norm(dim=1) ** 2 + 1e-8)
        x_special_refined_loop = (x_special_refined_loop - step_size_loop.view(-1, 1, 1, 1) * grad_lip_loop).detach().requires_grad_(True)
    
    #save the sample refined in a png
    out_png_special_loop = os.path.join(out_dir, 'hkr_specialfromnoise_loop.png')
    x_special_refined_loop = x_special_refined_loop[:16]
    plot_ims(out_png_special_loop, x_special_refined_loop)


     

    # do the same but from data
    x_special_data = x_data.clone().detach().requires_grad_(True)
    energy_lip_data = model(x_special_data)
    print('Energy Lip data:', energy_lip_data.mean().item())
    grad_lip_data = t.autograd.grad(energy_lip_data.sum(), [x_special_data])[0]
    step_size_data = (energy_lip_data) / (grad_lip_data.view(grad_lip_data.size(0), -1).norm(dim=1) ** 2 + 1e-8)
    x_special_data_refined = (x_special_data - step_size_data.view(-1, 1, 1, 1) * grad_lip_data).detach()

    out_png_special_data = os.path.join(out_dir, 'hkr_specialfromdata.png')
    x_special_data_refined = x_special_data_refined[:16]
    plot_ims(out_png_special_data, x_special_data_refined)

    #do the same but with a loop for 10 steps
    x_special_data_refined_loop = x_special_data.clone().detach().requires_grad_(True)
    for _ in range(10):
        energy_lip_data_loop = model(x_special_data_refined_loop)
        print('Energy Lip data loop:', energy_lip_data_loop.mean().item())
        grad_lip_data_loop = t.autograd.grad(energy_lip_data_loop.sum(), [x_special_data_refined_loop])[0]
        step_size_data_loop = (energy_lip_data_loop) / (grad_lip_data_loop.view(grad_lip_data_loop.size(0), -1).norm(dim=1) ** 2 + 1e-8)
        x_special_data_refined_loop = (x_special_data_refined_loop - step_size_data_loop.view(-1, 1, 1, 1) * grad_lip_data_loop).detach().requires_grad_(True)
    
    out_png_special_data_loop = os.path.join(out_dir, 'hkr_specialfromdata_loop.png')
    x_special_data_refined_loop = x_special_data_refined_loop[:16]
    plot_ims(out_png_special_data_loop, x_special_data_refined_loop)


if __name__ == '__main__':
    main()
