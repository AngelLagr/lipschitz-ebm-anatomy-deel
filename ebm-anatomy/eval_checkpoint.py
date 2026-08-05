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

EXP_DIR = './out_data/Expe/cifar10_semihalf10Lip_lang10_nofft_lr6e-6/'

# find any json config file in the code directory of the experiment
CONFIG_FILE = None
for file in os.listdir(EXP_DIR+'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR+'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}")

CHECKPOINT_FILE = EXP_DIR + 'checkpoints/net_017000.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 3000
LONGRUN_STEPS = 800
LONGRUN_INIT = 'uniform'  # 'uniform', 'gaussian', or 'fft'
FFT_NOISE_BANDS = ('low', 'mid', 'high')  # pour init_type='fft'
SEED = 42

# Calcule le FID toutes les FID_EVAL_INTERVAL itérations Langevin.
# Mettre à None pour désactiver (comportement original).
FID_EVAL_INTERVAL = 50   # ex: 50, 100, 200 …

# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
lip_const = None
# temperature from training
if config['epsilon'] > 0:
    temp = 1 * (config['epsilon'] ** 2) / 2
else:
    temp = 1


############################################
# ## INIT PERSISTENT FROM MNIST (always) ## #
############################################
def langevin_grad(f, x_s_t_0_l, config, temp_l, lip_const=None,
                  fid_eval_interval=None):
    """
    Returns
    -------
    x_s_t_l   : tensor  – état final du Langevin
    ens       : (LONGRUN_STEPS, batch)  – énergie à chaque pas
    grads     : (LONGRUN_STEPS, batch)  – norme du gradient à chaque pas
    snapshots : dict {step (int) -> tensor cpu}
                clés = multiples de fid_eval_interval (+ étape finale)
                valeur = état x à ce pas, sur CPU
    """
    if WConstrained_enabled:
        f.Lip_const = lip_const

    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    actual_batch_size = x_s_t_0_l.shape[0]
    grads = t.zeros(LONGRUN_STEPS, actual_batch_size)
    ens   = t.zeros(LONGRUN_STEPS, actual_batch_size)

    snapshots = {}  # {step -> cpu tensor}

    # iterative langevin updates of MCMC samples
    for ell in range(LONGRUN_STEPS):
        f_val = f(x_s_t_l)
        en = f_val / temp
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += - ((config['epsilon']**2)/2) * grad + config['epsilon'] * t.randn_like(x_s_t_l)
        grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu() * ((config['epsilon']**2)/2)

        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')

        # ── snapshot FID ──────────────────────────────────────────────────
        if fid_eval_interval is not None:
            step_num = ell + 1   # 1-indexé pour que "step 100" = 100 iters faites
            if step_num % fid_eval_interval == 0 or step_num == LONGRUN_STEPS:
                snapshots[step_num] = x_s_t_l.detach().cpu()
        # ─────────────────────────────────────────────────────────────────

    return x_s_t_l.detach(), ens, grads, snapshots


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

def sample_s_t(f, config, device, L, init_type, batch_size, lip_const=None,
               fid_eval_interval=None):
    """
    Returns
    -------
    samples          : (batch_size, C, H, W) tensor sur `device`
    grad_results     : (LONGRUN_STEPS, batch_size) tensor
    agg_snapshots    : dict {step -> (batch_size, C, H, W) tensor CPU}
                       présent uniquement si fid_eval_interval n'est pas None
    """
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0 = sample_freq_noise(batch_size, band='mixed')
    else:
        raise RuntimeError('Invalid init_type for sample_s_t (use "uniform", "gaussian", or "fft")')

    results      = []
    grad_results = []
    # Clé : step (int) → liste de tenseurs CPU (un par chunk)
    chunk_snapshots = {}

    for i in range(0, batch_size, 1000):
        print(f"\n batch chunk :{i} \n")
        chunk = x_s_t_0[i:i+1000]
        out, _, grads, snaps = langevin_grad(
            f, chunk, config, temp, lip_const,
            fid_eval_interval=fid_eval_interval
        )
        results.append(out.cpu())
        grad_results.append(grads)

        # Accumule les snapshots de ce chunk
        for step, snap in snaps.items():
            chunk_snapshots.setdefault(step, []).append(snap)

        t.cuda.empty_cache()

    # Concatène les snapshots de tous les chunks
    agg_snapshots = {
        step: t.cat(tensors, dim=0)
        for step, tensors in chunk_snapshots.items()
    }

    return (t.cat(results, dim=0).to(device),
            t.cat(grad_results, dim=1),
            agg_snapshots)


# -------- build model --------
print('Setting up network and optimizer...')
# set up network
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
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
    logits_extractor = None
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
    logits_extractor = InceptionLogits().half().to(device)   

else:
    feature_extractor = None

with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)

# -------- compute long-run FID from noise init --------
x_s_t_longrun, longrun_grads, longrun_snapshots = sample_s_t(
    f=f,
    config=config,
    device=device,
    L=LONGRUN_STEPS,
    init_type=LONGRUN_INIT,
    batch_size=FID_BATCH_SIZE,
    lip_const=lip_const,
    fid_eval_interval=FID_EVAL_INTERVAL,
)

longrun_grads = longrun_grads.detach()

# -----------------------------------------------------------------------
# Calcul du FID sur chaque snapshot (dans l'ordre des steps)
# -----------------------------------------------------------------------
fid_steps  = sorted(longrun_snapshots.keys())
fid_values = []
is_values  = []

print(f"\n── Calcul du FID sur {len(fid_steps)} snapshots ──")
for step in fid_steps:
    snap = longrun_snapshots[step].to(device)
    fid_val = compute_fid_batch(x_q_fid, snap, feature_extractor,
                                config['data'] == 'cifar10')
    fid_values.append(fid_val)
    print(f"  Step {step:>5d} / {LONGRUN_STEPS}  →  FID = {fid_val:.4f}")
    del snap
    t.cuda.empty_cache()

if logits_extractor is not None:
    print(f"\n── Calcul de l'IS sur {len(fid_steps)} snapshots ──")
    for step in fid_steps:
        snap = longrun_snapshots[step].to(device)
        is_val = compute_is_batch(snap, logits_extractor, config['data'] == 'cifar10')
        is_values.append(is_val)
        print(f"  Step {step:>5d} / {LONGRUN_STEPS}  →  IS  = {is_val:.4f}")
        del snap
        t.cuda.empty_cache()

# -----------------------------------------------------------------------
# Sauvegarde des résultats FID intermédiaires en CSV
# -----------------------------------------------------------------------
df_fid_langevin = pd.DataFrame({'langevin_step': fid_steps, 'fid': fid_values})
df_fid_langevin.to_csv(EXP_DIR + 'longrun/fid_vs_langevin_steps.csv', index=False)
print(f"\nFID intermédiaires sauvegardés → {EXP_DIR}longrun/fid_vs_langevin_steps.csv")

if is_values:
    df_is_langevin = pd.DataFrame({'langevin_step': fid_steps, 'is_score': is_values})
    df_is_langevin.to_csv(EXP_DIR + 'longrun/is_vs_langevin_steps.csv', index=False)
    print(f"IS intermédiaires sauvegardés → {EXP_DIR}longrun/is_vs_langevin_steps.csv")

# -----------------------------------------------------------------------
# Double courbe : norme gradient  +  FID
# -----------------------------------------------------------------------
steps_axis = range(1, LONGRUN_STEPS + 1)
mean_grads = longrun_grads.mean(dim=1).cpu().numpy()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

line1, = ax1.plot(steps_axis, mean_grads, color='steelblue', linewidth=1.0,
                  alpha=0.85, label='Mean gradient norm')
ax1.set_xlabel('Langevin step')
ax1.set_ylabel('Mean gradient norm', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

line2, = ax2.plot(fid_steps, fid_values, color='darkorange', linewidth=1.5,
                  marker='o', markersize=4, label=f'FID (every {FID_EVAL_INTERVAL} steps)')
ax2.set_ylabel('FID', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

lines  = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', framealpha=0.85)

plt.title(f'Langevin dynamics — Gradient norm & FID\n'
          f'(batch={FID_BATCH_SIZE}, {LONGRUN_STEPS} steps, FID every {FID_EVAL_INTERVAL} steps, init={LONGRUN_INIT})')
plt.tight_layout()
out_path = EXP_DIR + 'longrun/langevin_grads_fid.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Double courbe sauvegardée → {out_path}")

if is_values:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    line1, = ax1.plot(steps_axis, mean_grads, color='steelblue', linewidth=1.0,
                      alpha=0.85, label='Mean gradient norm')
    ax1.set_xlabel('Langevin step')
    ax1.set_ylabel('Mean gradient norm', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    line2, = ax2.plot(fid_steps, is_values, color='seagreen', linewidth=1.5,
                      marker='o', markersize=4, label='IS score')
    ax2.set_ylabel('IS score', color='seagreen')
    ax2.tick_params(axis='y', labelcolor='seagreen')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.85)

    plt.title(f'Langevin dynamics — Gradient norm & IS\n'
              f'(batch={FID_BATCH_SIZE}, {LONGRUN_STEPS} steps, init={LONGRUN_INIT})')
    plt.tight_layout()
    out_path = EXP_DIR + 'longrun/langevin_grads_is.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Double courbe IS sauvegardée → {out_path}")

# -----------------------------------------------------------------------
# Visualisation & FID final (comportement original conservé)
# -----------------------------------------------------------------------
x_s_t_longrun_vis = x_s_t_longrun[:16]
plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample.png', x_s_t_longrun_vis)
plot_ims(EXP_DIR + 'longrun/' + 'q_sample.png', x_q_fid[:16])

fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor,
                                config['data'] == 'cifar10')
is_score = None
if logits_extractor is not None:
    is_score = compute_is_batch(x_s_t_longrun, logits_extractor,
                                config['data'] == 'cifar10')

print(EXP_DIR)
print(CHECKPOINT_FILE)
print(f'FID_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {fid_longrun:.6f}')
if is_score is not None:
    print(f'IS_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {is_score:.6f}')

# -----------------------------------------------------------------------
# Courbe norme gradient seule (comportement original conservé)
# -----------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(mean_grads)
plt.xlabel('Langevin step')
plt.ylabel('Mean gradient norm')
plt.tight_layout()
plt.savefig(EXP_DIR + 'longrun/' + 'langevin_grads.png', dpi=150)
plt.close()

# -----------------------------------------------------------------------
# Analyse CSV entraînement (comportement original conservé)
# -----------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

iters = df['iteration'].to_numpy()
fids  = df['FID'].to_numpy()

fids_no_nan  = fids[~pd.isna(fids)]
iters_no_nan = iters[~pd.isna(fids)]
top_k        = min(3, len(fids_no_nan))
best_idx     = np.argsort(fids_no_nan)[:top_k]
best_fids    = fids_no_nan[best_idx]
best_iters   = iters_no_nan[best_idx]
best_fid     = best_fids[0]
best_iter    = best_iters[0]

print(f'Top {top_k} FID points:')
for i in range(top_k):
    print(f'Iter={best_iters[i]:.0f}, FID={best_fids[i]:.6f}')

idxfirst = 0
idx10    = next(i for i, v in enumerate(fids) if v <= 100)

x1, y1 = float(iters[idxfirst]), float(fids[idxfirst])
x2, y2 = float(iters[idx10]),    float(fids[idx10])

a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

print(f'First point: iter={x1:.0f}, fid={y1:.6f}')
print(f'Point FID<=10: iter={x2:.0f}, fid={y2:.6f}')
print(f'Droite: f(x) = a*x + b avec a={a:.8f}, b={b:.8f}')