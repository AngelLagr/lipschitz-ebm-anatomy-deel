import json
import os
import random

import matplotlib.pyplot as plt
import torch as t
import torch.fft as fft
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as tr
from utils import*
from nets import *


EXP_DIR = './out_data/Expe/hkrtest/cifar10_lip_lang10_HKR_REAL_alpha0999/'
BATCH_SIZE = 200
REFINE_STEPS = 10
SEED = 42
FFT_NOISE_BANDS = ('low', 'mid', 'high')
NOISE_INIT = 'checkpoint'  # 'fft' or 'uniform' or 'checkpoint'


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


def load_checkpoint(checkpoint_path):
    return t.load(checkpoint_path, map_location=device)


def load_model(config, checkpoint):
    model = build_model(config)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
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


def sample_init_noise(config, batch_size, noise_init=NOISE_INIT):
    if noise_init == 'fft':
        return sample_freq_noise(config, batch_size)
    if noise_init == 'uniform':
        return 2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device) - 1
    raise RuntimeError('Invalid noise init type (use "fft" or "uniform")')


def sample_checkpoint_noise(checkpoint, config, batch_size):
    if not (isinstance(checkpoint, dict) and 's_t_0' in checkpoint):
        raise RuntimeError('Checkpoint does not contain s_t_0')
    buffer = checkpoint['s_t_0'].to(device)
    if buffer.shape[0] < batch_size:
        raise RuntimeError(f's_t_0 buffer too small: has {buffer.shape[0]} samples, needs {batch_size}')
    inds = t.randperm(buffer.shape[0], device=device)[:batch_size]
    return buffer[inds]


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


def louis_refine(model, x0, iters=5,noise_eps=0.1):
    x = x0.clone().detach().requires_grad_(True)

    for _ in range(iters):
        energy_lip = model(x)
        grad_lip = t.autograd.grad(energy_lip.sum(), [x])[0]
        step_size = (noise_eps*t.randn_like(energy_lip) + energy_lip) / (grad_lip.view(grad_lip.size(0), -1).norm(dim=1) ** 2 + 1e-8)
        x = (x - step_size.view(-1, 1, 1, 1) * grad_lip).detach().requires_grad_(True)


    return x.detach()


def main():
    config_path = find_config(EXP_DIR)
    ckpt_path = find_checkpoint(EXP_DIR)

    with open(config_path) as f:
        config = json.load(f)

    checkpoint = load_checkpoint(ckpt_path)
    model = load_model(config, checkpoint)
    x_data = sample_data_batch(config, BATCH_SIZE)
    if NOISE_INIT == 'checkpoint':
        x_noise = sample_checkpoint_noise(checkpoint, config, BATCH_SIZE)
    else:
        x_noise = sample_init_noise(config, BATCH_SIZE, noise_init=NOISE_INIT)
    x_refined = langevin_refine(model, x_noise, config['epsilon'], REFINE_STEPS)
    
    #plot ims the xrefined
    out_dir = os.path.join(EXP_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    plot_ims(os.path.join(out_dir, 'x_refined.png'), x_refined[:64])


    x_louis_fromnoise = louis_refine(model, x_noise)
    x_louis_fromdata = louis_refine(model, x_data)

    with t.no_grad():
        hkr_data = model(x_data).view(-1).cpu().numpy()
        hkr_noise = model(x_noise).view(-1).cpu().numpy()
        hkr_refined = model(x_refined).view(-1).cpu().numpy()
        hkr_louis_fromnoise = model(x_louis_fromnoise).view(-1).cpu().numpy()
        hkr_louis_fromdata = model(x_louis_fromdata).view(-1).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(hkr_data, bins=50, alpha=0.5, label='data')
    #plt.hist(hkr_noise, bins=50, alpha=0.5, label='noise')
    plt.hist(hkr_refined, bins=50, alpha=0.5, label='noise refined shortrun')
    #plt.hist(hkr_louis_fromnoise, bins=50, alpha=0.5, label='noise louis step')
    #plt.hist(hkr_louis_fromdata, bins=50, alpha=0.5, label='data louis step')
    plt.legend()
    plt.title('HKR values histogram (data / noise / noise refined)')
    plt.tight_layout()
    out_png = os.path.join(out_dir, 'hkr_hist_simple.png')
    plt.savefig(out_png, dpi=150)

    alphas = t.linspace(0.0, 1.0, 51, device=device)
    interp_energy = []
    with t.no_grad():
        #xdata to xlouis fromdata
        for alpha in alphas:
            x_alpha = (1.0 - alpha) * x_data + alpha * x_louis_fromdata
            interp_energy.append(model(x_alpha).view(-1).mean().item())

        #xlouis fromdata to xlouis fromnoise
        for alpha in alphas:
            x_alpha = (1.0 - alpha) * x_louis_fromdata + alpha * x_louis_fromnoise
            interp_energy.append(model(x_alpha).view(-1).mean().item())
        #xlouis fromnoise to xnoise
        for alpha in alphas:
            x_alpha = (1.0 - alpha) * x_louis_fromnoise + alpha * x_noise
            interp_energy.append(model(x_alpha).view(-1).mean().item())

    interp_png = os.path.join(out_dir, 'hkr_interpolation_curve.png')
    plt.figure(figsize=(6, 4))
    x_axis = list(range(1, len(interp_energy) + 1))
    plt.plot(x_axis, interp_energy, linewidth=2)
    plt.xlabel('sample index')
    plt.ylabel('f(x_alpha)')
    plt.title('Interpolation entre data -> louis_fromdata -> louis_fromnoise -> noise')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(interp_png, dpi=150)

    interp_frames = []
    with t.no_grad():
        # data to xlouis fromdata
        x_start = x_data[:1]
        x_end = x_louis_fromdata[:1]
        for alpha in alphas[::5]:
            interp_frames.append((1.0 - alpha) * x_start + alpha * x_end)

        # xlouis fromdata to xlouis fromnoise
        x_start = x_louis_fromdata[:1]
        x_end = x_louis_fromnoise[:1]
        for alpha in alphas[::5]:
            interp_frames.append((1.0 - alpha) * x_start + alpha * x_end)
        
        # xlouis fromnoise to xnoise
        x_start = x_louis_fromnoise[:1]
        x_end = x_noise[:1]
        for alpha in alphas[::5]:
            interp_frames.append((1.0 - alpha) * x_start + alpha * x_end)
    interp_frames = t.cat(interp_frames, dim=0)

    interp_imgs_png = os.path.join(out_dir, 'hkr_interpolation_frames_line.png')
    tv.utils.save_image(t.clamp(interp_frames, -1., 1.), interp_imgs_png, normalize=True, nrow=interp_frames.shape[0])

    print('Checkpoint:', ckpt_path)
    print('Saved:', out_png)
    print('Saved:', interp_png)
    print('Saved:', interp_imgs_png)
    print('Energy noise:', hkr_noise.mean().item())
    print('Energy data:', hkr_data.mean().item())

    # Try to do a special refinement: one Louis step and plot before/after.
    print('Energy Lip noise step:', hkr_louis_fromnoise.mean().item())
    plot_ims(os.path.join(out_dir, 'hkr_specialfromnoise_before.png'), x_noise[:64])
    plot_ims(os.path.join(out_dir, 'hkr_specialfromnoise.png'), x_louis_fromnoise[:64])

    print('Energy Lip data step:', hkr_louis_fromdata.mean().item())
    plot_ims(os.path.join(out_dir, 'hkr_specialfromdata_before.png'), x_data[:64])
    plot_ims(os.path.join(out_dir, 'hkr_specialfromdata.png'), x_louis_fromdata[:64])
   

if __name__ == '__main__':
    main()
