#####################################################################
# ## EVAL CASCADE COARSE-TO-FINE (multi-resolution EBM cascade)   ##
#####################################################################

import os
import json
import random
import numpy as np
import pandas as pd
import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from ebm_utils import *
from nets import *
from utilfid import *

EXP_DIR = './out_data/Expe/cifar10_lang10_dim4x8x16x32/'

# Must match the resolutions used to train the checkpoint being loaded.
CASCADE_RESOLUTIONS = [2,4,8, 16, 32]

# Longrun Langevin steps for the very first stage (noise -> CASCADE_RESOLUTIONS[0])
L_STEPS_STAGE0 = 200

# Longrun Langevin steps for every subsequent transition (len == len(CASCADE_RESOLUTIONS) - 1)
CASCADE_STEPS = [15, 10, 7, 7]


# Leave at None to load the latest full checkpoint
# (checkpoints/last_checkpoint.pth), which contains every stage of the
# cascade together with the recorded lip_const.
#
# Set a specific iteration number (e.g. 5000) to load the weights saved
# at that iteration, via the files net_XXXXXX.pth (stage 0) and
# net_resR_XXXXXX.pth (subsequent stages, R = resolution), which are the
# ones written by train_data.py every `log_freq` iterations.
CHECKPOINT_ITER = 10000   # e.g. 5000

CKPT_DIR = EXP_DIR + 'checkpoints/'
CHECKPOINT_FILE = CKPT_DIR + 'last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'

# find the config json saved during training
CONFIG_FILE = None
for file in os.listdir(EXP_DIR + 'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR + 'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}code/")

FID_BATCH_SIZE = 3000

# init of the very first stage (pure noise)
STAGE0_INIT = 'uniform'  # 'uniform', 'gaussian', or 'fft'
FFT_NOISE_BANDS = ('low', 'mid', 'high')

# compute an intermediate FID every N steps inside each stage.
# One value per resolution in CASCADE_RESOLUTIONS (len must match).
# Use None for a given stage to disable intermediate FID there and only
# compute its final FID (useful for very cheap/low-res stages, or to
# save time on stages you don't care to inspect in detail).
#
# Example matching CASCADE_RESOLUTIONS = [2, 4, 8, 16, 32]:
#   CASCADE_FID_EVAL_INTERVAL = [None, None, 10, 10, 5]
CASCADE_FID_EVAL_INTERVAL = [50, 2, 2, 2, 1]
 
N_IMAGES_PLOT = 16  # number of images in the saved grids
SEED = 42
 
# the output folder includes the chosen iteration so results from
# different evals don't overwrite each other
_ckpt_tag = f'iter_0{CHECKPOINT_ITER:06d}' if CHECKPOINT_ITER is not None else 'last'
OUT_DIR = EXP_DIR + f'eval_cascade/{_ckpt_tag}/'
os.makedirs(OUT_DIR, exist_ok=True)
 
 
# =========================
# SETUP
# =========================
 
with open(CONFIG_FILE) as file:
    config = json.load(file)
 
# cascade validation
assert CASCADE_RESOLUTIONS[-1] == config['im_sz'], \
    'The last value of CASCADE_RESOLUTIONS must equal config["im_sz"] ({}).'.format(config['im_sz'])
assert len(CASCADE_STEPS) == len(CASCADE_RESOLUTIONS) - 1, \
    'CASCADE_STEPS must contain exactly len(CASCADE_RESOLUTIONS)-1 values (one per transition).'
for a, b in zip(CASCADE_RESOLUTIONS[:-1], CASCADE_RESOLUTIONS[1:]):
    assert a < b, 'CASCADE_RESOLUTIONS must be strictly increasing ({} >= {}).'.format(a, b)
assert len(CASCADE_FID_EVAL_INTERVAL) == len(CASCADE_RESOLUTIONS), \
    'CASCADE_FID_EVAL_INTERVAL must contain exactly one value per resolution in CASCADE_RESOLUTIONS ' \
    '(len={} expected, got {}).'.format(len(CASCADE_RESOLUTIONS), len(CASCADE_FID_EVAL_INTERVAL))
 
N_LEVELS = len(CASCADE_RESOLUTIONS)
STAGE_LABELS = ['Stage {} ({}x{})'.format(idx, res, res) for idx, res in enumerate(CASCADE_RESOLUTIONS)]
 
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
 
WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
 
t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
 
 
# =========================
# BUILD THE N_LEVELS NETWORKS (same logic as train_data.py)
# =========================
 
net_bank = {'bigtest': BigTestModel, 'semihalf': Semihalf, 'test': TestModel, 'vanilla': VanillaNet,
            'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet, 'vanillaHalfLip': VanillaNetHalfLip,
            'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet': Wide_ResNet,
            'wideresnetLip': Wide_ResNet_LLip, 'wideresnetLip_Logsumexp': Wide_ResNet_LLip_Logsumexp,
            'orthogonium': VanillaNetOrthogonium}
 
 
def build_net():
    if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip", "orthogonium", "test", "testinv", "semihalf", "bigtest"):
        net = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
    elif config['net_type'] == "wideresnet22":
        net = net_bank["wideresnet"](n_c=config['im_ch'], depth=22).to(device)
    elif config['net_type'] == "wideresnet34":
        net = net_bank["wideresnet"](n_c=config['im_ch'], depth=34).to(device)
    elif config['net_type'] == "wideresnetLip22":
        net = net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'], depth=22).to(device)
    else:
        net = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
    return net
 
 
def list_available_checkpoints():
    """List the iteration numbers available, inferred from the net_XXXXXX.pth
    files present in the checkpoints/ folder (stage 0 weights)."""
    iters = []
    if not os.path.isdir(CKPT_DIR):
        return iters
    for fname in os.listdir(CKPT_DIR):
        if fname.startswith('net_') and fname.endswith('.pth') and not fname.startswith('net_res'):
            middle = fname[len('net_'):-len('.pth')]
            if middle.isdigit():
                iters.append(int(middle))
    return sorted(iters)
 
 
print('Building {} cascade networks: {}'.format(N_LEVELS, ' -> '.join(str(r) for r in CASCADE_RESOLUTIONS)))
nets = [build_net() for _ in range(N_LEVELS)]
 
lip_const = None
 
if CHECKPOINT_ITER is None:
    # -------- load the latest full checkpoint --------
    print(f'Loading latest full checkpoint: {CHECKPOINT_FILE}')
    checkpoint = t.load(CHECKPOINT_FILE, map_location=device)
 
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        nets[0].load_state_dict(checkpoint['model_state_dict'])
        for lvl in range(1, N_LEVELS):
            res = CASCADE_RESOLUTIONS[lvl]
            key = 'model_res{}_state_dict'.format(res)
            if key not in checkpoint:
                raise RuntimeError(
                    'Checkpoint missing key "{}" for resolution {}. '
                    'Does CASCADE_RESOLUTIONS match the resolutions used to train this checkpoint?'.format(key, res))
            nets[lvl].load_state_dict(checkpoint[key])
        if WConstrained_enabled:
            lip_const = float(checkpoint['lip_const_record'][-1])
        print(f"Checkpoint loaded (iter={checkpoint.get('iter', '?')}).")
    else:
        raise RuntimeError('This script expects a FULL checkpoint (last_checkpoint.pth) '
                            'containing model_state_dict / model_res{R}_state_dict for every stage.')
else:
    # -------- load the weights saved at a specific iteration --------
    available = list_available_checkpoints()
    if CHECKPOINT_ITER not in available:
        raise FileNotFoundError(
            f"No checkpoint found for iteration {CHECKPOINT_ITER} in {CKPT_DIR}.\n"
            f"Available iterations: {available}"
        )
 
    print(f'Loading checkpoint at iter {CHECKPOINT_ITER} :')
    net_file = CKPT_DIR + 'net_{:>06d}.pth'.format(CHECKPOINT_ITER)
    print(f'  stage 0 ({CASCADE_RESOLUTIONS[0]}x{CASCADE_RESOLUTIONS[0]}) : {net_file}')
    nets[0].load_state_dict(t.load(net_file, map_location=device))
 
    for lvl in range(1, N_LEVELS):
        res = CASCADE_RESOLUTIONS[lvl]
        net_res_file = CKPT_DIR + 'net_res{}_{:>06d}.pth'.format(res, CHECKPOINT_ITER)
        print(f'  stage {lvl} ({res}x{res}) : {net_res_file}')
        nets[lvl].load_state_dict(t.load(net_res_file, map_location=device))
 
    # lip_const is not saved separately in these partial checkpoints
    print(f"Checkpoints loaded (iter={CHECKPOINT_ITER}).")
 
for net in nets:
    net.eval()
    net.to(device)
t.cuda.empty_cache()
 
 
# =========================
# REAL REFERENCE DATA (one per resolution, for FID)
# =========================
 
data = {
    'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
    'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
    'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
}
 
norm = tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch'])))
 
 
def build_level_transform(res, is_final):
    """Transform pipeline for a given cascade resolution.
    - intermediate levels: downsample to `res` then blocky nearest-neighbour
      upsampling back to config['im_sz'].
    - last level: image at its true final resolution, no blockiness."""
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
        norm,
    ])
 
 
print('Loading reference datasets ({})...'.format(', '.join('{}x{}'.format(r, r) for r in CASCADE_RESOLUTIONS)))
level_data_ref = []
for idx, res in enumerate(CASCADE_RESOLUTIONS):
    is_final = (idx == N_LEVELS - 1)
    transform_res = build_level_transform(res, is_final)
    ds = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform_res)]).to(device)
    level_data_ref.append(ds)
 
 
def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]
 
 
with t.no_grad():
    x_q_fid_per_level = []
    for ds in level_data_ref:
        x_fid = sample_image_set(ds, FID_BATCH_SIZE) + config['data_epsilon'] * t.randn(
            FID_BATCH_SIZE, *ds.shape[1:], device=device)
        x_q_fid_per_level.append(x_fid)
 
# -------- FID feature extractor --------
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None
 
if feature_extractor is None:
    raise RuntimeError(f"No FID feature extractor available for data='{config['data']}'.")
 
 
# =========================
# FREQUENCY NOISE (for STAGE0_INIT='fft')
# =========================
 
def make_freq_mask(sz, band, device):
    y = t.fft.fftfreq(sz, device=device).reshape(-1, 1)
    x = t.fft.fftfreq(sz, device=device).reshape(1, -1)
    r2 = x ** 2 + y ** 2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04 ** 2))
    elif band == 'mid':
        low = t.exp(-r2 / (2 * 0.04 ** 2))
        high = t.exp(-r2 / (2 * 0.15 ** 2))
        return high - low
    elif band == 'high':
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
    freq = t.fft.fft2(noise)
    filtered = t.empty_like(noise)
 
    for current_band in FFT_NOISE_BANDS:
        current_inds = [idx for idx, band_name in enumerate(bands) if band_name == current_band]
        if not current_inds:
            continue
        mask = make_freq_mask(config['im_sz'], current_band, device)[None, None, :, :]
        filtered[current_inds] = t.fft.ifft2(freq[current_inds] * mask).real
 
    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
 
 
# =========================
# GENERIC LANGEVIN FOR A SINGLE CASCADE STAGE
# =========================
 
def langevin_stage(net, x_init, L, fid_eval_interval=None):
    """
    Runs Langevin dynamics with `net`, starting from `x_init` (already at
    the right pixel resolution, no noise added on top of the init: either
    it is pure noise (stage 0), or it is the output of the previous stage).
 
    Returns:
      x_final    : final state (device)
      mean_grads : (L,) numpy array, mean gradient norm at each step
      snapshots  : {step -> cpu tensor} for intermediate FID computation
    """
    x = t.autograd.Variable(x_init.clone(), requires_grad=True)
    grads = t.zeros(L, x.shape[0])
    snapshots = {}
 
    for ell in range(L):
        f_val = net(x)
        f_prime = t.autograd.grad(f_val.sum(), [x])[0]
        x.data += -f_prime + config['epsilon'] * t.randn_like(x)
        grads[ell] = f_prime.view(f_prime.shape[0], -1).norm(dim=1).cpu()
 
        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == L:
            print('  step {} / {}   mean grad={:.6f}'.format(ell + 1, L, grads[ell].mean()), end='\r')
 
        if fid_eval_interval is not None:
            step_num = ell + 1
            if step_num % fid_eval_interval == 0 or step_num == L:
                snapshots[step_num] = x.detach().cpu()
    print()
 
    return x.detach(), grads.mean(dim=1).numpy(), snapshots
 
 
def run_stage_chunked(net, x_init, L, fid_eval_interval=None, chunk_size=1000):
    """Splits the batch into chunks to limit GPU memory usage, same idea as eval.py."""
    results, snaps_agg = [], {}
    grads_all = []
    n = x_init.shape[0]
    for i in range(0, n, chunk_size):
        print(f'\n batch chunk : {i}')
        chunk = x_init[i:i + chunk_size]
        out, grads, snaps = langevin_stage(net, chunk, L, fid_eval_interval=fid_eval_interval)
        results.append(out.cpu())
        grads_all.append(grads)
        for step, snap in snaps.items():
            snaps_agg.setdefault(step, []).append(snap)
        t.cuda.empty_cache()
 
    x_final = t.cat(results, dim=0).to(device)
    mean_grads = np.mean(np.stack(grads_all, axis=0), axis=0) if grads_all else np.array([])
    snaps_agg = {step: t.cat(tensors, dim=0) for step, tensors in snaps_agg.items()}
    return x_final, mean_grads, snaps_agg
 
 
def fid_over_snapshots(snapshots, x_real_ref):
    steps = sorted(snapshots.keys())
    fids = []
    for step in steps:
        snap = snapshots[step].to(device)
        fid_val = compute_fid_batch(x_real_ref, snap, feature_extractor, config['data'] == 'cifar10')
        fids.append(fid_val)
        print(f'  Step {step:>5d}  ->  FID = {fid_val:.4f}')
        del snap
        t.cuda.empty_cache()
    return steps, fids
 
 
# =========================
# CASCADE GENERATION ── generic loop over every stage
# =========================
 
x_stage_outputs = [None] * N_LEVELS   # final samples per stage
grads_per_stage = [None] * N_LEVELS   # mean gradient norm curve per stage
snaps_per_stage = [None] * N_LEVELS   # FID snapshots per stage
 
print('\n=== Stage 0 : noise -> {}x{} (via net 0) ==='.format(CASCADE_RESOLUTIONS[0], CASCADE_RESOLUTIONS[0]))
if STAGE0_INIT == 'uniform':
    x0 = (2 * t.rand([FID_BATCH_SIZE, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
elif STAGE0_INIT == 'gaussian':
    x0 = t.randn([FID_BATCH_SIZE, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
elif STAGE0_INIT == 'fft':
    x0 = sample_freq_noise(FID_BATCH_SIZE, band='mixed')
else:
    raise RuntimeError('Invalid STAGE0_INIT (uniform / gaussian / fft)')
 
x_out, grads_out, snaps_out = run_stage_chunked(
    nets[0], x0, L_STEPS_STAGE0, fid_eval_interval=CASCADE_FID_EVAL_INTERVAL[0])
x_stage_outputs[0] = x_out
grads_per_stage[0] = grads_out
snaps_per_stage[0] = snaps_out
 
for lvl in range(1, N_LEVELS):
    res_prev = CASCADE_RESOLUTIONS[lvl - 1]
    res_lvl = CASCADE_RESOLUTIONS[lvl]
    steps_lvl = CASCADE_STEPS[lvl - 1]
    print('\n=== Stage {} : {}x{} -> {}x{} (via net {}) ==='.format(lvl, res_prev, res_prev, res_lvl, res_lvl, lvl))
 
    x_out, grads_out, snaps_out = run_stage_chunked(
        nets[lvl], x_stage_outputs[lvl - 1], steps_lvl, fid_eval_interval=CASCADE_FID_EVAL_INTERVAL[lvl])
    x_stage_outputs[lvl] = x_out
    grads_per_stage[lvl] = grads_out
    snaps_per_stage[lvl] = snaps_out
 
 
# =========================
# FID PER RESOLUTION
# =========================
 
print('\n=== Final FID per stage ===')
fid_per_stage = []
for lvl in range(N_LEVELS):
    fid_val = compute_fid_batch(x_q_fid_per_level[lvl], x_stage_outputs[lvl], feature_extractor,
                                 config['data'] == 'cifar10')
    fid_per_stage.append(fid_val)
    is_final = (lvl == N_LEVELS - 1)
    tag = 'real' if is_final else 'blocky'
    print('{} ({}x{} {})   FID = {:.4f}'.format(STAGE_LABELS[lvl], CASCADE_RESOLUTIONS[lvl],
                                                 CASCADE_RESOLUTIONS[lvl], tag, fid_val))
 
pd.DataFrame({
    'stage': list(range(N_LEVELS)),
    'resolution': CASCADE_RESOLUTIONS,
    'FID': fid_per_stage,
}).to_csv(OUT_DIR + 'fid_per_stage.csv', index=False)
 
# -------- intermediate FID (as a function of the Langevin step), per stage --------
steps_per_stage = [None] * N_LEVELS
fidcurve_per_stage = [None] * N_LEVELS
for lvl in range(N_LEVELS):
    print('\n=== Intermediate FID — {} ==='.format(STAGE_LABELS[lvl]))
    steps_lvl, fidcurve_lvl = fid_over_snapshots(snaps_per_stage[lvl], x_q_fid_per_level[lvl])
    steps_per_stage[lvl] = steps_lvl
    fidcurve_per_stage[lvl] = fidcurve_lvl
    res = CASCADE_RESOLUTIONS[lvl]
    pd.DataFrame({'step': steps_lvl, 'fid': fidcurve_lvl}).to_csv(
        OUT_DIR + 'fid_vs_step_stage{}_{}x{}.csv'.format(lvl, res, res), index=False)
 
 
# =========================
# PLOTS
# =========================
 
# -------- generated images per stage --------
for lvl in range(N_LEVELS):
    res = CASCADE_RESOLUTIONS[lvl]
    plot_ims(OUT_DIR + 'gen_stage{}_{}x{}.png'.format(lvl, res, res), x_stage_outputs[lvl][:N_IMAGES_PLOT])
 
# -------- real reference images (same resolutions) for visual comparison --------
for lvl in range(N_LEVELS):
    res = CASCADE_RESOLUTIONS[lvl]
    plot_ims(OUT_DIR + 'real_stage{}_{}x{}.png'.format(lvl, res, res), x_q_fid_per_level[lvl][:N_IMAGES_PLOT])
 
# -------- summary grid : cascade progression for the same samples --------
n_show = min(8, N_IMAGES_PLOT)
fig, axes = plt.subplots(N_LEVELS, n_show, figsize=(2 * n_show, 2 * N_LEVELS))
if N_LEVELS == 1:
    axes = axes.reshape(1, -1)
for row in range(N_LEVELS):
    imgs = x_stage_outputs[row]
    title = STAGE_LABELS[row]
    imgs_np = ((imgs[:n_show].detach().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
    for col in range(n_show):
        ax = axes[row, col]
        img = imgs_np[col]
        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(title, rotation=0, labelpad=40, fontsize=10)
plt.suptitle('Cascade progression (same samples through {})'.format(' -> '.join(str(r) for r in CASCADE_RESOLUTIONS)))
plt.tight_layout()
plt.savefig(OUT_DIR + 'cascade_progression.png', dpi=150)
plt.close()
 
# -------- mean gradient curves per stage --------
fig, axes = plt.subplots(1, N_LEVELS, figsize=(5 * N_LEVELS, 4), sharey=False)
if N_LEVELS == 1:
    axes = [axes]
for ax, grads, title in zip(axes, grads_per_stage, STAGE_LABELS):
    ax.plot(grads, color='steelblue')
    ax.set_xlabel('Langevin step')
    ax.set_ylabel('Mean gradient norm')
    ax.set_title(title)
plt.tight_layout()
plt.savefig(OUT_DIR + 'gradient_norm_per_stage.png', dpi=150)
plt.close()
 
# -------- FID vs step curve, one color per stage --------
plt.figure(figsize=(10, 5))
for lvl in range(N_LEVELS):
    res = CASCADE_RESOLUTIONS[lvl]
    plt.plot(steps_per_stage[lvl], fidcurve_per_stage[lvl], marker='o', markersize=3,
              label='{} vs real {}x{}'.format(STAGE_LABELS[lvl], res, res))
plt.xlabel('Langevin step (within stage)')
plt.ylabel('FID')
plt.title('FID vs Langevin step per stage (batch={}, stage0 init={})'.format(FID_BATCH_SIZE, STAGE0_INIT))
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + 'fid_vs_step_all_stages.png', dpi=150)
plt.close()
 
# -------- final FID barplot per resolution --------
plt.figure(figsize=(2 * N_LEVELS + 2, 4))
bar_labels = ['{}x{}\n(net {})'.format(res, res, lvl) for lvl, res in enumerate(CASCADE_RESOLUTIONS)]
plt.bar(bar_labels, fid_per_stage, color=plt.cm.viridis(np.linspace(0.2, 0.8, N_LEVELS)))
plt.ylabel('Final FID')
plt.title('Final FID per cascade resolution')
plt.tight_layout()
plt.savefig(OUT_DIR + 'fid_final_per_stage.png', dpi=150)
plt.close()
 
print(f'\nEverything was saved in: {OUT_DIR}')
print(f'  - gen_stage*_*.png / real_stage*_*.png : generated/real images per resolution')
print(f'  - cascade_progression.png : same samples through all {N_LEVELS} stages')
print(f'  - fid_per_stage.csv, fid_final_per_stage.png : final FID per resolution')
print(f'  - fid_vs_step_stage*_*.csv, fid_vs_step_all_stages.png : FID as a function of the Langevin step')
print(f'  - gradient_norm_per_stage.png : mean gradient norm per stage')
 