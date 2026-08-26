import json
import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch.fft as fft

from ebm_utils import*
from nets import *
from utilfid import *

# =========================
# PARAMETRES (EN HAUT)
# =========================

EXP_DIR = './out_data/Expe/cifar10_Langevin10_only_negative_forebm_regul_0001/'

# find any json config file in the code directory of the experiment
CONFIG_FILE = None
for file in os.listdir(EXP_DIR+'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR+'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}")

CHECKPOINT_FILE = EXP_DIR + 'checkpoints/net_099000.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 3000
LONGRUN_STEPS = 2000
HKR_THRESHOLD = 0.5
TAU = 0.2
BIG_HEAD_NONLIP =False

LONGRUN_INIT = 'fft'  # 'uniform', 'gaussian', or 'fft'
FFT_NOISE_BANDS = ('low', 'mid', 'high')
SEED = 42

# Calcule le FID toutes les FID_EVAL_INTERVAL itérations Langevin.
# Mettre à None pour désactiver (comportement original).
FID_EVAL_INTERVAL = 100   # ex: 50, 100, 200 …


# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
lip_const = None


# -----------------------------------------------------------------------
# langevin_grad  – capture des snapshots toutes les fid_eval_interval iters
# -----------------------------------------------------------------------
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

    for ell in range(LONGRUN_STEPS):
        f_val = f(x_s_t_l)
        en    = f_val
        ens[ell] = en.detach().cpu().squeeze(-1)
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += - grad + config['epsilon'] * t.randn_like(x_s_t_l)
        # Calcul de l'énergie HKR pour le suivi sur une petite portion du batch (pour éviter de saturer la mémoire GPU)
        x_s_t_l_batch = x_s_t_l[:min(100, actual_batch_size)]
        hkr_energy = f.hkr_head(f.features(x_s_t_l_batch)).detach().cpu().squeeze(-1)
        grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()

        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f} HKR={:>14.9f}'.format(
                ell+1, LONGRUN_STEPS,
                ens[ell].mean(), grads[ell].mean(), hkr_energy.mean()), end='\r')

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
        low  = t.exp(-r2 / (2 * 0.04**2))
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

    noise    = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device)
    freq     = fft.fft2(noise)
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


# -----------------------------------------------------------------------
# sample_s_t  – agrège les snapshots de tous les chunks
# -----------------------------------------------------------------------
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
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'],
                                config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'],
                            config['im_sz'], config['im_sz']]).to(device)
    elif init_type == 'fft':
        x_s_t_0 = sample_freq_noise(batch_size, band='mixed')
    else:
        raise RuntimeError('Invalid init_type (use "uniform", "gaussian", or "fft")')

    results      = []
    grad_results = []
    # Clé : step (int) → liste de tenseurs CPU (un par chunk)
    chunk_snapshots = {}

    for i in range(0, batch_size, 1000):
        print(f"\n batch chunk :{i} \n")
        chunk = x_s_t_0[i:i+1000]
        out, _, grads, snaps = langevin_grad(
            f, chunk, config, 1.0, lip_const,
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


# -----------------------------------------------------------------------
# Modèle
# -----------------------------------------------------------------------
def load_checkpoint(model, checkpoint_path):
    checkpoint = t.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'lip_const_record' in checkpoint and len(checkpoint['lip_const_record']) > 0:
            model.lip_const = float(checkpoint['lip_const_record'][-1])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, checkpoint


class Identity(nn.Module):
    def forward(self, x):
        return x


def conv3x3_lip(in_planes, out_planes, stride=1):
    return torchlip.SpectralConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True,
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

class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(x + out)

class DualHeadLipEBM(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, n_classes=10,
                 lip_const=1.0, hkr_switch_threshold=0.0, hkr_tau=0.1):
        super().__init__()
        self.lip_const = float(lip_const)
        self.hkr_switch_threshold = float(hkr_switch_threshold)
        self.hkr_tau = float(hkr_tau)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.activation  = torchlip.GroupSort2()
        self.lipavgpool2d = torchlip.ScaledAvgPool2d(kernel_size=8)
        self.conv1  = conv3x3_lip(n_c, n_stages[0])
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
            layers += [torchlip.SpectralLinear(in_dim, hidden_dim), torchlip.GroupSort2(),
                       torchlip.SpectralLinear(hidden_dim, hidden_dim), torchlip.GroupSort2(),
                       torchlip.SpectralLinear(hidden_dim, out_dim), torchlip.GroupSort2(),
                       torchlip.SpectralLinear(out_dim, 1)]
        else:
            if BIG_HEAD_NONLIP:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.05))
                layers += [ResMLPBlock(hidden_dim) for _ in range(3)]
                layers.append(nn.Linear(hidden_dim, out_dim))
                layers.append(nn.LeakyReLU(0.05))
                layers.append(nn.Linear(out_dim, 1))

            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.05))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.05))
                layers.append(nn.Linear(hidden_dim, out_dim))
                layers.append(nn.LeakyReLU(0.05))
                layers.append(nn.Linear(out_dim, 1))
        return nn.Sequential(*layers)

    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
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
        return self.ebm_head(h), self.lip_const * self.hkr_head(h)

    def forward(self, x, head='theta'):
        ebm_energy, hkr_energy = self.energy_components(x)
        if head in ('lip', 'hkr'):
            return hkr_energy
        if head in ('nonlip', 'ebm'):
            return ebm_energy
        if head == 'theta':
            alpha = t.sigmoid((hkr_energy - self.hkr_switch_threshold) / self.hkr_tau).detach()
            return alpha * hkr_energy + (1.0 - alpha) * ebm_energy
        raise RuntimeError('Invalid head (use "theta", "hkr" or "ebm")')


# -----------------------------------------------------------------------
# Chargement du checkpoint
# -----------------------------------------------------------------------
f = DualHeadLipEBM(
    depth=config.get('depth', 16),
    widen_factor=config.get('widen_factor', 2),
    n_c=config['im_ch'],
    n_classes=config.get('n_classes', 10),
    lip_const=config.get('Lip_const', 1.0),
    hkr_switch_threshold=HKR_THRESHOLD,
    hkr_tau=TAU,
).to(device)

f, checkpoint = load_checkpoint(f, CHECKPOINT_FILE)

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

# -----------------------------------------------------------------------
# Chargement des données
# -----------------------------------------------------------------------
data = {
    'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
    'mnist':   lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
    'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
}
transform = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
])
q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)

# -----------------------------------------------------------------------
# Feature extractor pour le FID
# -----------------------------------------------------------------------
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
    logits_extractor = InceptionLogits().half().to(device)

else:
    feature_extractor = None

with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)

# -----------------------------------------------------------------------
# Long-run Langevin  +  snapshots pour FID parallèle
# -----------------------------------------------------------------------
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
fid_steps  = sorted(longrun_snapshots.keys())   # ex: [100, 200, …, 2000]
fid_values = []

print(f"\n── Calcul du FID sur {len(fid_steps)} snapshots ──")
for step in fid_steps:
    snap = longrun_snapshots[step].to(device)   # ramène sur GPU pour Inception
    fid_val = compute_fid_batch(x_q_fid, snap, feature_extractor,
                                config['data'] == 'cifar10')
    fid_values.append(fid_val)
    print(f"  Step {step:>5d} / {LONGRUN_STEPS}  →  FID = {fid_val:.4f}")
    del snap
    t.cuda.empty_cache()

# -----------------------------------------------------------------------
# Sauvegarde des résultats FID intermédiaires en CSV
# -----------------------------------------------------------------------
df_fid_langevin = pd.DataFrame({'langevin_step': fid_steps, 'fid': fid_values})
df_fid_langevin.to_csv(EXP_DIR + 'longrun/fid_vs_langevin_steps.csv', index=False)
print(f"\nFID intermédiaires sauvegardés → {EXP_DIR}longrun/fid_vs_langevin_steps.csv")

# -----------------------------------------------------------------------
# Double courbe : norme gradient  +  FID
# -----------------------------------------------------------------------
steps_axis = range(1, LONGRUN_STEPS + 1)
mean_grads = longrun_grads.mean(dim=1).cpu().numpy()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

# Courbe 1 – norme moyenne du gradient (axe gauche, bleu)
line1, = ax1.plot(steps_axis, mean_grads, color='steelblue', linewidth=1.0,
                  alpha=0.85, label='Mean gradient norm')
ax1.set_xlabel('Langevin step')
ax1.set_ylabel('Mean gradient norm', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Courbe 2 – FID (axe droit, orange) avec marqueurs
line2, = ax2.plot(fid_steps, fid_values, color='darkorange', linewidth=1.5,
                  marker='o', markersize=4, label=f'FID (every {FID_EVAL_INTERVAL} steps)')
ax2.set_ylabel('FID', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

# Légende unifiée
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

# -----------------------------------------------------------------------
# Visualisation & FID final (comportement original conservé)
# -----------------------------------------------------------------------
x_s_t_longrun_vis = x_s_t_longrun[:16]
plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample.png', x_s_t_longrun_vis)
plot_ims(EXP_DIR + 'longrun/' + 'q_sample.png', x_q_fid[:16])

fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor,
                                config['data'] == 'cifar10')
is_score = compute_is_batch(x_s_t_longrun, logits_extractor,
                            config['data'] == 'cifar10')

print(EXP_DIR)
print(CHECKPOINT_FILE)
print(f'FID_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {fid_longrun:.6f}')
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
df    = pd.read_csv(CSV_FILE)
iters = df['iteration'].to_numpy()
fids  = df['FID'].to_numpy()

fids_no_nan  = fids[~pd.isna(fids)]
iters_no_nan = iters[~pd.isna(fids)]
top_k        = min(3, len(fids_no_nan))
best_idx     = np.argsort(fids_no_nan)[:top_k]
best_fids    = fids_no_nan[best_idx]
best_iters   = iters_no_nan[best_idx]

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