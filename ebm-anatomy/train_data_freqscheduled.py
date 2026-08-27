##########################################################
# ## TRAIN EBM WITH FREQUENCY SHEDULING IN THE BUFFER ## #
##########################################################

import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
import csv
import numpy as np
import random
import torch.fft as fft
from nets import *
from ebm_utils import download_flowers_data, plot_ims, plot_diagnostics
from utilfid import *

# ── experiment paths ────────────────────────────────────────────────────────────
EXP_DIR = './out_data/Expe/cifar10_lang10_semihalfLip_freqscheduled_smooth/'
CONFIG_FILE = './config_locker/cifar10_Langevin10_semihalfLipDu3FFT.json'

# ── sampling hyper-parameters ───────────────────────────────────────────────────
RANDOMRATE               = 0.95   # probability of reusing a persistent sample vs. fresh noise
LANGEVIN_SQUARED_NORMALISED = False  # normalise gradient by its squared norm
DST_SCALED               = False  # rescale CD loss by 2/ε²

# ── FFT frequency-band schedule ─────────────────────────────────────────────────
#   iteration 0          → [low=1.0, mid=0.0, high=0.0]  (pure low-frequency)
#   iteration SCHED_END  → [low=1/3, mid=1/3, high=1/3]  (uniform mix)
FFT_NOISE_BANDS = ('low', 'mid', 'high')
SCHED_END_ITER = 70_000

#######################
# ## INITIAL SETUP ## #
#######################

with open(CONFIG_FILE) as file:
    config = json.load(file)

# create experiment directory (ask before overwriting)
if os.path.exists(EXP_DIR):
    choice = input(
        'Experiment folder "{}" already exists. Overwrite it? (y/n) '.format(EXP_DIR)
    )
    if choice.lower() == 'y':
        import shutil
        shutil.rmtree(EXP_DIR)
    else:
        raise RuntimeError(
            'Folder "{}" already exists. Use a different EXP_DIR.'.format(EXP_DIR)
        )
os.makedirs(EXP_DIR)
for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
    os.mkdir(EXP_DIR + folder)


def lerp(a, b, t):
    return a + (b - a) * t

def save_code():
    for file_name in ['train_data.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        with open('./' + file_name, 'r') as fin, \
             open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w') as fout:
            fout.write(fin.read())

save_code()

# reproducibility
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

device    = t.device('cuda' if t.cuda.is_available() else 'cpu')
start_iter = 0

# temperature scaling for DST_SCALED mode
temp = (config['epsilon'] ** 2) / 2 if DST_SCALED else 1.0

# FID feature extractor
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor(
        './fidcheckpoints/mnist_feature_extractor.pth', device
    )
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None


########################
# ## TRAINING SETUP ## #
########################

print('Setting up network and optimizer...')
# set up network
net_bank = {'semihalf': Semihalf, 'test' : TestModel,'testinv': TestInvModel, 'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet':Wide_ResNet, "wideresnetLip" : Wide_ResNet_LLip, "wideresnetLip_Logsumexp": Wide_ResNet_LLip_Logsumexp, "orthogonium": VanillaNetOrthogonium}
if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip", "orthogonium","test","testinv", "semihalf"):
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

optim_bank = {'adam': t.optim.Adam, 'sgd': t.optim.SGD, 'adamW': t.optim.AdamW}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale LR to match Langevin noise variance
    if DST_SCALED:
        config['lr_init'] *= (config['epsilon'] ** 2) / 2
        config['lr_min']  *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])


print('Processing data...')
if config['data'] == 'flowers':
    download_flowers_data()

data_loaders = {
    'cifar10': lambda path, fn: datasets.CIFAR10(root=path, transform=fn, download=True),
    'mnist':   lambda path, fn: datasets.MNIST(root=path,   transform=fn, download=True),
    'flowers': lambda path, fn: datasets.ImageFolder(root=path, transform=fn),
}
transform = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    tr.Normalize(
        tuple(0.5 * t.ones(config['im_ch'])),
        tuple(0.5 * t.ones(config['im_ch']))
    ),
])
q = t.stack([
    x[0] for x in data_loaders[config['data']]('./data/' + config['data'], transform)
]).to(device)
print('Dataset size:', q.shape[0])


#################################
# ## FFT NOISE & SCHEDULE    ## #
#################################

def make_freq_mask(sz: int, band: str, dev: t.device) -> t.Tensor:
    """
    Returns a 2-D Gaussian-based mask in frequency space for the given band.
      low  : σ = 0.04  (DC neighbourhood)
      mid  : annular region between σ=0.04 and σ=0.15
      high : everything outside σ=0.15
    """
    fy = fft.fftfreq(sz, device=dev).reshape(-1, 1)
    fx = fft.fftfreq(sz, device=dev).reshape(1, -1)
    r2 = fx ** 2 + fy ** 2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04 ** 2))
    elif band == 'mid':
        return t.exp(-r2 / (2 * 0.15 ** 2)) - t.exp(-r2 / (2 * 0.04 ** 2))
    elif band == 'high':
        return 1.0 - t.exp(-r2 / (2 * 0.15 ** 2))
    raise RuntimeError('Invalid band (use "low", "mid", or "high")')

def get_band_weights(iteration: int):
    """
    Schedule:

        0 -----------> 30k -----------> 60k 
      (1,0,0)       (0.8,0.2,0)     (1/3,1/3,1/3)    

    puis reste constant.
    """

    if iteration <= SCHED_END_ITER:

        half = SCHED_END_ITER / 2

        if iteration <= half:
            t = iteration / half

            start = (1.0, 0.0, 0.0)
            end   = (0.8, 0.2, 0.0)

        else:
            t = (iteration - half) / half

            start = (0.8, 0.2, 0.0)
            end   = (1/3, 1/3, 1/3)

    else:
        return [1/3, 1/3, 1/3]  

    w_low  = lerp(start[0], end[0], t)
    w_mid  = lerp(start[1], end[1], t)
    w_high = lerp(start[2], end[2], t)

    # sécurité numérique
    s = w_low + w_mid + w_high
    return [w_low / s, w_mid / s, w_high / s]

def sample_freq_noise(batch_size: int, iteration: int = 0) -> t.Tensor:
    """
    Generate a batch of FFT-filtered noise images.

    Each sample's frequency band is drawn i.i.d. from the categorical
    distribution defined by get_band_weights(iteration):
      - iteration=0          → every sample is low-frequency
      - iteration=SCHED_END  → uniform mix of low / mid / high

    Output is normalised to [-1, 1] per sample.
    """
    weights     = t.tensor(get_band_weights(iteration), dtype=t.float32)
    band_indices = t.multinomial(
        weights.unsqueeze(0).expand(batch_size, -1), num_samples=1
    ).squeeze(1)  # shape: (batch_size,)
    bands = [FFT_NOISE_BANDS[idx] for idx in band_indices.tolist()]

    C, H = config['im_ch'], config['im_sz']
    noise    = t.randn([batch_size, C, H, H], device=device)
    freq     = fft.fft2(noise)
    filtered = t.empty_like(noise)

    for band in FFT_NOISE_BANDS:
        inds = [k for k, b in enumerate(bands) if b == band]
        if not inds:
            continue
        mask = make_freq_mask(H, band, device)[None, None, :, :]  # (1,1,H,H)
        filtered[inds] = fft.ifft2(freq[inds] * mask).real

    # normalise each sample independently to [-1, 1]
    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


# ── persistent replay buffer ─────────────────────────────────────────────────────
# Initialised at iteration=0, so every entry is pure low-frequency noise.
s_t_0 = sample_freq_noise(q.shape[0], iteration=0)


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

def sample_image_set(image_set: t.Tensor, batch_size: int = config['batch_size']):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds], rand_inds


def sample_q(batch_size: int = config['batch_size']) -> t.Tensor:
    """Sample a mini-batch from the data distribution (with a tiny noise jitter)."""
    x_q = sample_image_set(q, batch_size=batch_size)[0]
    return x_q + config['data_epsilon'] * t.randn_like(x_q)


def sample_s_t(
    L: int,
    init_type: str,
    iteration: int = 0,
    update_s_t_0: bool = True,
    batch_size: int = config['batch_size'],
):
    """
    Run L steps of Langevin MCMC starting from `init_type` initialisation.

    For 'persistent' init, the RANDOMRATE fraction of samples come from the
    replay buffer; the remainder are freshly sampled with the current frequency
    schedule (so early training = low-freq injections, late = mixed).
    """
    # ── initialise MCMC states ────────────────────────────────────────────────
    if init_type == 'persistent':
        mask    = t.rand(batch_size, device=device) < RANDOMRATE
        indices = t.randperm(s_t_0.shape[0], device=device)[:batch_size]

        persistent_samples = s_t_0[indices]
        # fresh samples follow the current frequency schedule
        fresh_samples      = sample_freq_noise(batch_size, iteration=iteration)

        x_s_t_0   = t.where(mask.view(-1, 1, 1, 1), persistent_samples, fresh_samples)
        s_t_0_inds = indices

    elif init_type == 'data':
        x_s_t_0, s_t_0_inds = sample_q(batch_size=batch_size), None

    elif init_type == 'uniform':
        C, H = config['im_ch'], config['im_sz']
        x_s_t_0   = (2 * t.rand([batch_size, C, H, H]) - 1).to(device)
        s_t_0_inds = None

    elif init_type == 'gaussian':
        C, H = config['im_ch'], config['im_sz']
        x_s_t_0   = t.randn([batch_size, C, H, H]).to(device)
        s_t_0_inds = None

    else:
        raise RuntimeError(
            'Invalid init_type (use "persistent", "data", "uniform", or "gaussian")'
        )

    # ── Langevin updates ───────────────────────────────────────────────────────
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1, device=device)

    for _ in range(L):
        f_val   = f(x_s_t)
        f_prime = t.autograd.grad(f_val.sum(), [x_s_t])[0]

        if LANGEVIN_SQUARED_NORMALISED:
            denom   = f_prime.view(f_prime.shape[0], -1).norm(dim=1).pow(2).view(-1, 1, 1, 1) + 1e-8
            step    = f_prime / denom
        else:
            step = f_prime

        x_s_t.data += -step + config['epsilon'] * t.randn_like(x_s_t)
        r_s_t       += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

    if init_type == 'persistent' and update_s_t_0:
        s_t_0.data[s_t_0_inds] = x_s_t.detach().data.clone()

    return x_s_t.detach(), r_s_t.squeeze() / L


def _langevin_longrun(
    model, x_init: t.Tensor, cfg: dict, temperature: float, batch_size: int
):
    """Inner long-run Langevin loop (used for FID evaluation)."""
    x = t.autograd.Variable(x_init.clone(), requires_grad=True)
    grads = t.zeros(cfg['num_longrun_steps'], batch_size)
    ens   = t.zeros(cfg['num_longrun_steps'], batch_size)

    for ell in range(cfg['num_longrun_steps']):
        f_val = model(x)
        en    = f_val / temperature
        ens[ell] = en.detach().cpu()

        grad = t.autograd.grad(en.sum(), [x])[0]

        if LANGEVIN_SQUARED_NORMALISED:
            denom = grad.view(grad.shape[0], -1).norm(dim=1).pow(2).view(-1, 1, 1, 1) + 1e-8
            step  = grad / denom
        else:
            step = grad

        if DST_SCALED:
            x.data += -((cfg['epsilon'] ** 2) / 2) * step + cfg['epsilon'] * t.randn_like(x)
            grads[ell] = ((cfg['epsilon'] ** 2) / 2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        else:
            x.data += -step + cfg['epsilon'] * t.randn_like(x)
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()

    return x.detach(), ens, grads


def sample_s_tlongrun(model, it: int, cfg: dict, dev: t.device, init_type: str, batch_size: int) -> t.Tensor:
    """
    Generate long-run MCMC samples for FID evaluation.
    Always uses the terminal frequency mix (iteration=it)
    to initialise, regardless of where training currently is.
    """
    C, H = cfg['im_ch'], cfg['im_sz']
    if init_type == 'uniform':
        x0 = (2 * t.rand([batch_size, C, H, H]) - 1).to(dev)
    elif init_type == 'gaussian':
        x0 = t.randn([batch_size, C, H, H]).to(dev)
    elif init_type == 'fft':
        # use terminal uniform mix for evaluation
        x0 = sample_freq_noise(batch_size, iteration=it)
    else:
        raise RuntimeError('Invalid init_type (use "uniform", "gaussian", or "fft")')

    return _langevin_longrun(model, x0, cfg, temp, batch_size)[0]


#######################
# ## TRAINING LOOP ## #
#######################

d_s_t_record      = t.zeros(config['num_train_iters'], device=device)
r_s_t_record      = t.zeros(config['num_train_iters'], device=device)
fid_record        = t.full((config['num_train_iters'],), float('nan'), device=device)
instability_record = t.full((config['num_train_iters'],), float('nan'), device=device)
f_x_q_record      = t.zeros(config['num_train_iters'], device=device)
f_x_st_record     = t.zeros(config['num_train_iters'], device=device)

fid_score        = float('nan')
csv_file_path    = EXP_DIR + 'plots/training_metrics.csv'
last_csv_written = 0
stability_window = 2


def save_full_checkpoint(iteration_idx: int):
    iter_done = iteration_idx + 1
    t.save(
        {
            'iter':               iter_done,
            'model_state_dict':   f.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            's_t_0':              s_t_0.detach().cpu(),
            'd_s_t_record':       d_s_t_record[:iter_done].detach().cpu(),
            'r_s_t_record':       r_s_t_record[:iter_done].detach().cpu(),
            'fid_record':         fid_record[:iter_done].detach().cpu(),
            'instability_record': instability_record[:iter_done].detach().cpu(),
            'f_x_q_record':       f_x_q_record[:iter_done].detach().cpu(),
            'f_x_st_record':      f_x_st_record[:iter_done].detach().cpu(),
            'fid_score':          float(fid_score),
            'last_csv_written':   last_csv_written,
            'config':             config,
        },
        EXP_DIR + 'checkpoints/last_checkpoint.pth',
    )


def compute_instability_index(i: int) -> t.Tensor:
    if i + 1 < stability_window:
        return t.tensor(float('nan'), device=device)
    start  = i + 1 - stability_window
    eps    = 1e-8
    w_st   = f_x_st_record[start:i + 1]
    w_q    = f_x_q_record[start:i + 1]
    cv_st  = w_st.std(unbiased=False) / (w_st.abs().mean() + eps)
    cv_q   = w_q.std(unbiased=False)  / (w_q.abs().mean()  + eps)
    return 0.5 * (cv_st + cv_q)


print('Training has started.')
for i in range(start_iter, config['num_train_iters']):

    # ── positive samples (data distribution) ─────────────────────────────────
    x_q = sample_q()

    # ── negative samples (model distribution via short-run MCMC) ─────────────
    # `iteration=i` drives the frequency-band schedule for fresh buffer injections
    x_s_t, r_s_t = sample_s_t(
        L=config['num_shortrun_steps'],
        init_type=config['shortrun_init'],
        iteration=i,
    )

    # ── contrastive-divergence loss: E_data[f] − E_model[f] ──────────────────
    f_x_q  = f(x_q).mean()
    f_x_st = f(x_s_t).mean()
    loss   = f_x_q - f_x_st            # unscaled loss (recorded as-is)

    d_s_t  = loss * (2 / config['epsilon'] ** 2) if DST_SCALED else loss

    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # ── diagnostics ───────────────────────────────────────────────────────────
    d_s_t_record[i]       = loss.detach()
    r_s_t_record[i]       = r_s_t
    f_x_q_record[i]       = f_x_q.detach()
    f_x_st_record[i]      = f_x_st.detach()
    instability_record[i] = compute_instability_index(i)

    bw = get_band_weights(i)
    print(
        '{:>6d}  loss={:>12.6f}  r_s_t={:>12.6f}  instab={:>8.6f}  '
        'fid={:>8.2f}  bands=[L={:.3f} M={:.3f} H={:.3f}]'.format(
            i + 1,
            loss.detach().item(),
            r_s_t.item(),
            instability_record[i].item(),
            fid_score,
            bw[0], bw[1], bw[2],
        )
    )

    # ── FID evaluation ────────────────────────────────────────────────────────
    if feature_extractor is not None and (
        i == 0 or (i + 1) % config.get('log_freq', 100) == 0
    ):
        if i == 0:
            fid_record[i] = 500.0
        else:
            fid_batch = 1000
            x_q_fid   = sample_q(batch_size=fid_batch)
            x_st_fid  = sample_s_tlongrun(f, i, config, device, init_type='fft', batch_size=fid_batch)
            fid_score = compute_fid_batch(x_q_fid, x_st_fid, feature_extractor, config['data'] == 'cifar10')
            fid_score = min(fid_score, 500.0)
            fid_record[i] = t.tensor(fid_score, device=device)

    # ── learning-rate annealing ───────────────────────────────────────────────
    for pg in optim.param_groups:
        pg['lr'] = max(config['lr_min'], pg['lr'] * config['lr_decay'])

    # ── periodic logging & checkpointing ─────────────────────────────────────
    if (i + 1) % config['log_freq'] == 0 or (i + 1) == config['num_train_iters']:
        # CSV
        csv_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csv_fh:
            writer = csv.writer(csv_fh)
            if not csv_exists:
                writer.writerow(['iteration', 'FID', 'DST', 'RST', 'Instability', 'fxq', 'fxst',
                                  'band_w_low', 'band_w_mid', 'band_w_high'])
            for j in range(last_csv_written, i + 1):
                bwj = get_band_weights(j)
                writer.writerow([
                    j + 1,
                    float(fid_record[j].cpu()),
                    float(d_s_t_record[j].cpu()),
                    float(r_s_t_record[j].cpu()),
                    float(instability_record[j].cpu()),
                    float(f_x_q_record[j].cpu()),
                    float(f_x_st_record[j].cpu()),
                    round(bwj[0], 6),
                    round(bwj[1], 6),
                    round(bwj[2], 6),
                ])
        last_csv_written = i + 1

        # images
        plot_ims(EXP_DIR + 'shortrun/x_s_t_{:>06d}.png'.format(i + 1), x_s_t)
        if config['shortrun_init'] == 'persistent':
            plot_ims(
                EXP_DIR + 'shortrun/x_s_t_0_{:>06d}.png'.format(i + 1),
                s_t_0[:config['batch_size']],
            )

        # model weights & full checkpoint
        t.save(f.state_dict(), EXP_DIR + 'checkpoints/net_{:>06d}.pth'.format(i + 1))
        save_full_checkpoint(i)

        # diagnostic plots
        if (i + 1) > 1:
            plot_diagnostics(
                i,
                d_s_t_record, r_s_t_record, fid_record,
                f_x_q_record, f_x_st_record, instability_record,
                EXP_DIR + 'plots/',
            )

    # ── long-run diagnostic chains ────────────────────────────────────────────
    if config['log_longrun'] and (i + 1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples (L={} steps).'.format(
            i + 1, config['num_longrun_steps']
        ))
        x_p_theta = sample_s_t(
            L=config['num_longrun_steps'],
            init_type=config['longrun_init'],
            iteration=i,
            update_s_t_0=False,
        )[0]
        plot_ims(EXP_DIR + 'longrun/longrun_{:>06d}.png'.format(i + 1), x_p_theta)
        print('{:>6d}   Long-run samples saved.'.format(i + 1))