#######################################
# ## TRAIN EBM USING IMAGE DATASET ## #
#######################################

import csv
import json
import os
import random

import numpy as np
import torch as t
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as tr
from deel import torchlip

from utilfid import *
from utils import download_flowers_data, plot_diagnostics, plot_ims


# directory for experiment results
# -----------------------------------------------------------------------------
EXP_DIR = './out_data/Expe/cifar10_Langevin10_WARMUP/'
CONFIG_FILE = './config_locker/cifar10_Langevin10_LipDu3FFT.json'
RANDOMRATE = 0.95

# Lipschitz constraint settings
K_FACTOR = 2.0  # factor to scale the Lipschitz constant for the K constraint
ALPHA = 0.995  # transition smoothness

# FFT noise settings
FFT_NOISE = True
FFT_NOISE_BANDS = ('low', 'mid', 'high')

# Sigmoid param
HKR_THRESHOLD = 0.0
TAU = 1.0

HKR_ALPHA = 0.99 # Hkr loss alpha param

LAMBDA_HKR = 10.0 # Weight for HKR loss in the combined loss 
HKR_WARMUP_ITERS = 5000 # number iteration with the HKR only
# -----------------------------------------------------------------------------


#######################
# ## NETWORK DEFS    ##
#######################


class Identity(nn.Module):
    def forward(self, x):
        return x


def conv3x3_lip(in_planes, out_planes, stride=1):
    return torchlip.SpectralConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


class wide_basic_lip(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = Identity()
        self.lip_residual = torchlip.LipResidual()
        self.act = torchlip.GroupSort2()

        self.conv1 = torchlip.SpectralConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = Identity()
        self.conv2 = torchlip.SpectralConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                torchlip.SpectralConv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=True)
            )

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out = (out + self.lip_residual(out, self.shortcut(x))) / 2
        return out


class DualHeadLipEBM(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, n_classes=10, lip_const=1.0, hkr_switch_threshold=0.0, hkr_tau=0.1):
        super().__init__()
        self.lip_const = float(lip_const)
        self.hkr_switch_threshold = float(hkr_switch_threshold)
        self.hkr_tau = float(hkr_tau)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.activation = torchlip.GroupSort2()
        self.lipavgpool2d = torchlip.ScaledAvgPool2d(kernel_size=8)

        self.conv1 = conv3x3_lip(n_c, n_stages[0])
        self.layer1 = self._wide_layer(n_stages[1], n, stride=1)
        self.layer2 = self._wide_layer(n_stages[2], n, stride=2)
        self.layer3 = self._wide_layer(n_stages[3], n, stride=2)
        self.bn1 = Identity()

        self.hkr_head = self._make_head(n_stages[3], n_classes, lip=True)
        self.ebm_head = self._make_head(n_stages[3], n_classes, lip=False)

    def _make_head(self, in_dim, out_dim, lip):
        hidden_dim = in_dim * 2
        layers = []
        if lip:
            layers.append(torchlip.SpectralLinear(in_dim, hidden_dim))
            layers.append(torchlip.GroupSort2())
            layers.append(torchlip.SpectralLinear(hidden_dim, hidden_dim))
            layers.append(torchlip.GroupSort2())
            layers.append(torchlip.SpectralLinear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.05))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.05))
            layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(wide_basic_lip(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.lipavgpool2d(out)
        return out.view(out.size(0), -1)

    def energy_components(self, x):
        h = self.features(x)
        hkr_logits = self.hkr_head(h)
        ebm_logits = self.ebm_head(h)
        hkr_energy = self.lip_const * hkr_logits.logsumexp(1)
        ebm_energy = ebm_logits.logsumexp(1)
        return ebm_energy, hkr_energy

    def forward(self, x, head='theta'):
        ebm_energy, hkr_energy = self.energy_components(x)

        if head in ('lip', 'hkr'):
            return hkr_energy
        if head in ('nonlip', 'ebm'):
            return ebm_energy
        if head == 'theta':
            alpha = t.sigmoid((hkr_energy - self.hkr_switch_threshold) / self.hkr_tau)
            alpha = alpha.detach()
            Etheta = alpha * hkr_energy + (1.0 - alpha) * ebm_energy
            return Etheta
        raise RuntimeError('Invalid head (use "theta", "hkr" or "ebm")')


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)# make directory for saving results
if os.path.exists(EXP_DIR):
    choice = input('Experiment folder "{}" already exists. Do you want to overwrite it? (y/n) '.format(EXP_DIR))
    if choice.lower() == 'y':
        import shutil

        shutil.rmtree(EXP_DIR)
        os.makedirs(EXP_DIR)
        for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
            os.mkdir(EXP_DIR + folder)
    else:
        raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)


def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)

    script_name = os.path.basename(__file__)
    for file_name in [script_name, 'utils.py', 'utilfid.py', CONFIG_FILE]:
        save_file(file_name)


save_code()

# set seed for cpu and CUDA, get device
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])
start_iter = 0

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

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

f = DualHeadLipEBM(
    depth=config.get('depth', 16),
    widen_factor=config.get('widen_factor', 2),
    n_c=config['im_ch'],
    n_classes=config.get('n_classes', 10),
    lip_const=config.get('Lip_const', 1.0),
    hkr_switch_threshold=HKR_THRESHOLD,
    hkr_tau=TAU,
).to(device)

current_lip_const = None
WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)

def update_lip_const(iteration_idx, loss=None, grad=None):
    lip_const = f.lip_const
    if not WConstrained_enabled:
        return None

    if WConstrained_enabled and grad is not None and loss is not None:
        print(f"Current Lip_const: {lip_const:.4f}")

        new_lip = abs(loss)/(((grad**2)/lip_const + 1e-8) * config['num_shortrun_steps']) 
        new_lip = K_FACTOR * new_lip
        new_lip_const = ALPHA*lip_const + (1-ALPHA)*new_lip
    else:
        new_lip_const = lip_const

    f.lip_const = float(new_lip_const)
    return f.lip_const


optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD, 'adamW': t.optim.AdamW}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

print('Processing data...')
if config['data'] == 'flowers':
    download_flowers_data()

data = {
    'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
    'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
    'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
}

transform = tr.Compose(
    [
        tr.Resize(config['im_sz']),
        tr.CenterCrop(config['im_sz']),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
    ]
)
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

print('Dataset size:', q.shape[0])


def make_freq_mask(sz, band, device):
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


if FFT_NOISE:
    s_t_0 = sample_freq_noise(q.shape[0], band='mixed')
else:
    s_t_0 = 2 * t.rand_like(q) - 1


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################


def sample_image_set(image_set, batch_size=config['batch_size']):
    rand_inds = t.randperm(image_set.shape[0])[0:batch_size]
    return image_set[rand_inds], rand_inds


def sample_q(batch_size=config['batch_size']):
    x_q = sample_image_set(q, batch_size=batch_size)[0]
    return x_q + config['data_epsilon'] * t.randn_like(x_q)


def sample_s_t(L, init_type, langevin_head, update_s_t_0=True, batch_size=config['batch_size']):
    def sample_s_t_0():
        if init_type == 'persistent':
            mask = t.rand(batch_size, device=device) < RANDOMRATE
            indices = t.randperm(s_t_0.shape[0], device=device)[:batch_size]
            persistent_samples = s_t_0[indices]

            if FFT_NOISE:
                random_samples = sample_freq_noise(batch_size, band='mixed')
            else:
                random_samples = 2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device) - 1

            x_s_t_0 = t.where(mask.view(-1, 1, 1, 1), persistent_samples, random_samples)
            return x_s_t_0, indices.clone()

        if init_type == 'data':
            return sample_q(batch_size=batch_size), None

        if init_type == 'uniform':
            noise_image = 2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1
            return noise_image.to(device), None

        if init_type == 'gaussian':
            noise_image = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']])
            return noise_image.to(device), None

        raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    x_s_t_0, s_t_0_inds = sample_s_t_0()

    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).to(device)
    for _ in range(L):
        f_prime = t.autograd.grad(f(x_s_t, head=langevin_head).sum(), [x_s_t])[0]
        x_s_t.data += -f_prime + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

    if init_type == 'persistent' and update_s_t_0:
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / max(1, L)


def langevin_grad(model, x_s_t_0_l, cfg, temp, batch_size, head):
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    grads = t.zeros(cfg['num_longrun_steps'], batch_size)
    ens = t.zeros(cfg['num_longrun_steps'], batch_size)

    for ell in range(cfg['num_longrun_steps']):
        en = model(x_s_t_l, head=head) / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += -((cfg['epsilon'] ** 2) / 2) * grad + cfg['epsilon'] * t.randn_like(x_s_t_l)
        grads[ell] = ((cfg['epsilon'] ** 2) / 2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()

    return x_s_t_l.detach(), ens, grads


def sample_s_tlongrun(model, cfg, device, init_type, batch_size, head):
    if init_type == 'uniform':
        x_s_t_0_l = (2 * t.rand([batch_size, cfg['im_ch'], cfg['im_sz'], cfg['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0_l = t.randn([batch_size, cfg['im_ch'], cfg['im_sz'], cfg['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0_l = sample_freq_noise(batch_size, band='mixed')
    else:
        raise RuntimeError('Invalid method for "init_type" (use "uniform", "gaussian", or "fft")')

    x_s_t_l = langevin_grad(model, x_s_t_0_l.clone(), cfg, (cfg['epsilon'] ** 2) / 2, batch_size, head=head)[0]
    return x_s_t_l.detach()


#######################
# ## TRAINING LOOP ## #
#######################


num_train_iters = config['num_train_iters']

d_s_t_record = t.zeros(num_train_iters).to(device)
r_s_t_record = t.zeros(num_train_iters).to(device)
fid_record = t.zeros(num_train_iters).to(device)
fid_record[:] = float('nan')
instability_record = t.zeros(num_train_iters).to(device)
instability_record[:] = float('nan')
fid_score = float('nan')
lip_const_record = t.zeros(num_train_iters).to(device)
lip_const_record[:] = float('nan')
f_x_q_record = t.zeros(num_train_iters).to(device)
f_x_st_record = t.zeros(num_train_iters).to(device)

d_s_t_lip_record = t.zeros(num_train_iters).to(device)
d_s_t_nonlip_record = t.zeros(num_train_iters).to(device)

csv_file_path = EXP_DIR + 'plots/' + 'training_metrics.csv'
last_csv_written = 0
stability_window = 2


def save_full_checkpoint(iteration_idx, training_mode, energy_mode, current_steps):
    iter_done = iteration_idx + 1
    checkpoint = {
        'iter': iter_done,
        'training_mode': training_mode,
        'energy_mode': energy_mode,
        'num_shortrun_steps': current_steps,
        'model_state_dict': f.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        's_t_0': s_t_0.detach().cpu(),
        'd_s_t_record': d_s_t_record[:iter_done].detach().cpu(),
        'r_s_t_record': r_s_t_record[:iter_done].detach().cpu(),
        'fid_record': fid_record[:iter_done].detach().cpu(),
        'instability_record': instability_record[:iter_done].detach().cpu(),
        'f_x_q_record': f_x_q_record[:iter_done].detach().cpu(),
        'f_x_st_record': f_x_st_record[:iter_done].detach().cpu(),
        'd_s_t_lip_record': d_s_t_lip_record[:iter_done].detach().cpu(),
        'd_s_t_nonlip_record': d_s_t_nonlip_record[:iter_done].detach().cpu(),
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

    fxst_window = f_x_st_record[start : i + 1]
    fxq_window = f_x_q_record[start : i + 1]

    fxst_cv = fxst_window.std(unbiased=False) / (fxst_window.abs().mean() + eps)
    fxq_cv = fxq_window.std(unbiased=False) / (fxq_window.abs().mean() + eps)

    return 0.5 * (fxst_cv + fxq_cv)


print('Training has started.')
num_shortrun_steps = config['num_shortrun_steps']
hkr_loss_fn = HKRLoss(alpha=HKR_ALPHA, min_margin=1.0)

for i in range(start_iter, num_train_iters):
    training_mode = 'hkr' if i < HKR_WARMUP_ITERS else 'theta'

    x_q = sample_q()
    x_s_t, r_s_t = sample_s_t(
        L=num_shortrun_steps,
        init_type=config['shortrun_init'],
        langevin_head=training_mode,
    )

    ebm_q, hkr_q = f.energy_components(x_q)
    ebm_s, hkr_s = f.energy_components(x_s_t)

    

    f_x_q_hkr = hkr_q.mean()
    f_x_st_hkr = hkr_s.mean()

    kr_term = f_x_q_hkr - f_x_st_hkr

    hkr_input = t.cat([hkr_q, hkr_s], dim=0).reshape(-1, 1)
    hkr_target = t.cat([
        -t.ones_like(hkr_q),
        t.ones_like(hkr_s),
    ], dim=0).reshape(-1, 1)

    hkr_loss = hkr_loss_fn(hkr_input, hkr_target)





    f_x_q_ebm = ebm_q.mean()
    f_x_st_ebm = ebm_s.mean()
    ebm_loss = f_x_q_ebm - f_x_st_ebm

    if i < HKR_WARMUP_ITERS:
        loss_raw = hkr_loss
        loss_saved = hkr_loss.detach().data.clone()
    else:
        loss_raw = ebm_loss + LAMBDA_HKR*hkr_loss
        loss_saved = loss_raw.detach().data.clone()

    if config['epsilon'] > 0:
        d_s_t = loss_raw * 2 / (config['epsilon'] ** 2)
    else:
        d_s_t = loss_raw

    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    if WConstrained_enabled and i >= HKR_WARMUP_ITERS:
        current_lip_const = update_lip_const(i, loss=kr_term * r_s_t, grad=r_s_t)

    d_s_t_record[i] = loss_saved if loss_saved < 600 else float(600)
    d_s_t_lip_record[i] = hkr_loss.detach()
    d_s_t_nonlip_record[i] = ebm_loss.detach()
    r_s_t_record[i] = r_s_t

    if WConstrained_enabled:
        lip_const_record[i] = current_lip_const if current_lip_const is not None else float('nan')

    if i < HKR_WARMUP_ITERS:
        f_x_q_record[i] = f_x_q_hkr.detach()
        f_x_st_record[i] = f_x_st_hkr.detach()
    else:
        f_x_q_record[i] = f_x_q_ebm.detach()
        f_x_st_record[i] = f_x_st_ebm.detach()

    print(f"Iteration {i+1}: mode={training_mode}, f(x_q)={f_x_q_record[i]:.4f}, f(x_s_t)={f_x_st_record[i]:.4f}, fhkr(x_q)={f_x_q_hkr:.4f}, fhkr(x_s_t)={f_x_st_hkr:.4f}")
    instability_record[i] = compute_instability_index(i)

    if feature_extractor is not None and (i == 0 or (i + 1) % config.get('log_freq', 100) == 0):
        if i == 0:
            fid_record[i] = t.tensor(500).to(device)
        else:
            fid_batch_size = 1000
            x_q_fid = sample_q(batch_size=fid_batch_size)
            if FFT_NOISE:
                x_s_t_fid = sample_s_tlongrun(
                    f,
                    config,
                    device,
                    init_type='fft',
                    batch_size=fid_batch_size,
                    head='theta' if i >= HKR_WARMUP_ITERS else 'hkr',
                )
            else:
                x_s_t_fid = sample_s_tlongrun(
                    f,
                    config,
                    device,
                    init_type='uniform',
                    batch_size=fid_batch_size,
                    head='theta' if i >= HKR_WARMUP_ITERS else 'hkr',
                )
            fid_score = compute_fid_batch(x_q_fid, x_s_t_fid, feature_extractor, config['data'] == 'cifar10')
            if fid_score > 500:
                fid_score = 500
            fid_record[i] = t.tensor(fid_score).to(device)

    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    print(
        '{:>6d}   d_s_t={:>14.9f}   d_hkr={:>14.9f}   d_ebm={:>14.9f}   r_s_t={:>14.9f}   instability={:>14.9f} fid={:>14.9f}'.format(
            i + 1,
            d_s_t_record[i],
            d_s_t_lip_record[i],
            d_s_t_nonlip_record[i],
            r_s_t,
            instability_record[i],
            fid_score,
        )
    )
    if current_lip_const is not None:
        print('{:>6d}   Lip_const={:>14.9f}'.format(i + 1, current_lip_const))

    if ((i + 1) % config['log_freq'] == 0) or ((i + 1) == num_train_iters):
        csv_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            if not csv_exists:
                csv_writer.writerow(
                    [
                        'iteration',
                        'mode',
                        'FID',
                        'DST',
                        'DST_hkr',
                        'DST_ebm',
                        'RST',
                        'Instability',
                        'fxq',
                        'fxst',
                        'Lip_const',
                    ]
                )
            for j in range(last_csv_written, i + 1):
                mode_j = 'hkr' if j < HKR_WARMUP_ITERS else 'theta'
                csv_writer.writerow(
                    [
                        j + 1,
                        mode_j,
                        float(fid_record[j].detach().cpu()),
                        float(d_s_t_record[j].detach().cpu()),
                        float(d_s_t_lip_record[j].detach().cpu()),
                        float(d_s_t_nonlip_record[j].detach().cpu()),
                        float(r_s_t_record[j].detach().cpu()),
                        float(instability_record[j].detach().cpu()),
                        float(f_x_q_record[j].detach().cpu()),
                        float(f_x_st_record[j].detach().cpu()),
                        float(lip_const_record[j].detach().cpu()),
                    ]
                )
        last_csv_written = i + 1

        plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_{:>06d}.png'.format(i + 1), x_s_t)
        if config['shortrun_init'] == 'persistent':
            plot_ims(EXP_DIR + 'shortrun/' + 'x_s_t_0_{:>06d}.png'.format(i + 1), s_t_0[0 : config['batch_size']])

        t.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i + 1))
        save_full_checkpoint(
            i,
            training_mode=training_mode,
            energy_mode='theta' if i >= HKR_WARMUP_ITERS else 'hkr',
            current_steps=num_shortrun_steps,
        )

        if (i + 1) > 1:
            plot_diagnostics(
                i,
                d_s_t_record,
                r_s_t_record,
                fid_record,
                f_x_q_record,
                f_x_st_record,
                instability_record,
                EXP_DIR + 'plots/',
            )

    if config['log_longrun'] and (i + 1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i + 1, config['num_longrun_steps']))
        x_p_theta = sample_s_t(
            L=config['num_longrun_steps'],
            init_type=config['longrun_init'],
            langevin_head='theta' if i >= HKR_WARMUP_ITERS else 'hkr',
            update_s_t_0=False,
        )[0]
        plot_ims(EXP_DIR + 'longrun/' + 'longrun_{:>06d}.png'.format(i + 1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i + 1))
