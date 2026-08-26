#####################################################################
# ## EVAL CASCADE COARSE-TO-FINE (f 8x8 -> f16 16x16 -> f32 32x32) ##
#####################################################################
#
# Ce script évalue le modèle produit par train_coarsetofine.py.
# Il charge f / f16 / f32 depuis le checkpoint complet, génère des
# images en cascade (chaque étage part du résultat Langevin de
# l'étage précédent, PAS d'un nouveau bruit), et pour chaque
# résolution (8x8, 16x16, 32x32) :
#   - calcule le FID face à la vraie distribution de cette résolution
#   - sauvegarde une grille d'images
#   - (optionnel) trace FID / gradient en fonction du pas Langevin
#
# =========================
# PARAMETRES (EN HAUT)
# =========================

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

EXP_DIR = './out_data/Expe/cifar10_lang20_dim8x8up16x16up32x32/'

# checkpoint complet (contient f, f16 et f32 en même temps)
CHECKPOINT_FILE = EXP_DIR + 'checkpoints/last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'

# trouve le json de config sauvegardé pendant l'entraînement
CONFIG_FILE = None
for file in os.listdir(EXP_DIR + 'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR + 'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f"No JSON config file found in {EXP_DIR}code/")

# tailles utilisées à l'entraînement (doivent matcher train_coarsetofine.py)
ORIG_SZ = 8    # résolution de détail "coarse" (stage f)
MID_SZ = 16    # résolution de détail "mid"    (stage f16)
# la résolution finale (stage f32) est config['im_sz']

FID_BATCH_SIZE = 1000

# nombre de pas Langevin par étage (peuvent être différents de l'entraînement,
# on veut typiquement un "long-run" plus long pour bien évaluer chaque stage)
L_STEPS_STAGE1 = 50    # bruit          -> 8x8 (f)
L_STEPS_STAGE2 = 300    # sortie stage 1 -> 16x16 (f16)
L_STEPS_STAGE3 = 300    # sortie stage 2 -> 32x32 (f32)

# init du tout premier stage (bruit pur)
STAGE1_INIT = 'uniform'  # 'uniform', 'gaussian', ou 'fft'
FFT_NOISE_BANDS = ('low', 'mid', 'high')  # pour STAGE1_INIT='fft'

# calcule le FID intermédiaire tous les N pas à l'intérieur de chaque étage
# (None pour désactiver et ne calculer que le FID final de chaque étage)
FID_EVAL_INTERVAL = 50

N_IMAGES_PLOT = 16  # nombre d'images dans les grilles sauvegardées
SEED = 42

OUT_DIR = EXP_DIR + 'eval_cascade/'
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# SETUP
# =========================

with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)

t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# =========================
# CONSTRUCTION DES 3 RESEAUX (même logique que train_coarsetofine.py)
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


print('Building networks f, f16, f32...')
f = build_net()
f16 = build_net()
f32 = build_net()

# -------- charge le checkpoint complet --------
print(f'Loading checkpoint: {CHECKPOINT_FILE}')
checkpoint = t.load(CHECKPOINT_FILE, map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    f.load_state_dict(checkpoint['model_state_dict'])
    f16.load_state_dict(checkpoint['model16_state_dict'])
    f32.load_state_dict(checkpoint['model32_state_dict'])
    lip_const = float(checkpoint['lip_const_record'][-1]) if WConstrained_enabled else None
    print(f"Checkpoint chargé (iter={checkpoint.get('iter', '?')}).")
else:
    raise RuntimeError('Ce script attend un checkpoint COMPLET (last_checkpoint.pth) '
                        'contenant model_state_dict / model16_state_dict / model32_state_dict.')

for net in (f, f16, f32):
    net.eval()
    net.to(device)
t.cuda.empty_cache()


# =========================
# DONNEES REELLES DE REFERENCE (une par résolution, pour le FID)
# =========================

data = {
    'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
    'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
    'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
}

norm = tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch'])))

# stage 1 (f)   : positif = dataset downsamplé à ORIG_SZ puis remonté en blocs
transform_8 = tr.Compose([
    tr.Resize(ORIG_SZ),
    tr.Resize(config['im_sz'], interpolation=tr.InterpolationMode.NEAREST),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    norm,
])

# stage 2 (f16) : positif = dataset downsamplé à MID_SZ puis remonté en blocs
transform_16 = tr.Compose([
    tr.Resize(MID_SZ),
    tr.Resize(config['im_sz'], interpolation=tr.InterpolationMode.NEAREST),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    norm,
])

# stage 3 (f32) : positif = dataset à sa vraie résolution
transform_32 = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    norm,
])

print('Loading reference datasets (8x8, 16x16, 32x32)...')
q8_ref = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform_8)]).to(device)
q16_ref = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform_16)]).to(device)
q32_ref = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform_32)]).to(device)


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]


with t.no_grad():
    x_q8_fid = sample_image_set(q8_ref, FID_BATCH_SIZE) + config['data_epsilon'] * t.randn(
        FID_BATCH_SIZE, *q8_ref.shape[1:], device=device)
    x_q16_fid = sample_image_set(q16_ref, FID_BATCH_SIZE) + config['data_epsilon'] * t.randn(
        FID_BATCH_SIZE, *q16_ref.shape[1:], device=device)
    x_q32_fid = sample_image_set(q32_ref, FID_BATCH_SIZE) + config['data_epsilon'] * t.randn(
        FID_BATCH_SIZE, *q32_ref.shape[1:], device=device)

# -------- feature extractor pour le FID --------
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None

if feature_extractor is None:
    raise RuntimeError(f"Pas de feature extractor FID disponible pour data='{config['data']}'.")


# =========================
# BRUIT FREQUENTIEL (pour STAGE1_INIT='fft')
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
# LANGEVIN GENERIQUE POUR UN ETAGE DE LA CASCADE
# =========================

def langevin_stage(net, x_init, L, fid_eval_interval=None):
    """
    Fait tourner Langevin avec `net` en partant de `x_init` (déjà à la bonne
    résolution pixel, pas de bruit ajouté à l'init : soit c'est du vrai bruit
    (stage 1), soit c'est la sortie de l'étage précédent).

    Retourne :
      x_final    : état final (device)
      mean_grads : (L,) numpy, norme moyenne du gradient à chaque pas
      snapshots  : {step -> tensor cpu} pour le FID intermédiaire
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
    """Découpe en chunks pour limiter la mémoire GPU, comme dans eval.py."""
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
# GENERATION EN CASCADE
# =========================

print('\n=== STAGE 1 : bruit -> 8x8 (via f) ===')
if STAGE1_INIT == 'uniform':
    x0 = (2 * t.rand([FID_BATCH_SIZE, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
elif STAGE1_INIT == 'gaussian':
    x0 = t.randn([FID_BATCH_SIZE, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
elif STAGE1_INIT == 'fft':
    x0 = sample_freq_noise(FID_BATCH_SIZE, band='mixed')
else:
    raise RuntimeError('STAGE1_INIT invalide (uniform / gaussian / fft)')

x_stage1, grads1, snaps1 = run_stage_chunked(f, x0, L_STEPS_STAGE1, fid_eval_interval=FID_EVAL_INTERVAL)

print('\n=== STAGE 2 : sortie stage1 -> 16x16 (via f16) ===')
x_stage2, grads2, snaps2 = run_stage_chunked(f16, x_stage1, L_STEPS_STAGE2, fid_eval_interval=FID_EVAL_INTERVAL)

print('\n=== STAGE 3 : sortie stage2 -> 32x32 (via f32) ===')
x_stage3, grads3, snaps3 = run_stage_chunked(f32, x_stage2, L_STEPS_STAGE3, fid_eval_interval=FID_EVAL_INTERVAL)


# =========================
# FID PAR RESOLUTION
# =========================

print('\n=== FID final par étage ===')
fid_stage1 = compute_fid_batch(x_q8_fid, x_stage1, feature_extractor, config['data'] == 'cifar10')
fid_stage2 = compute_fid_batch(x_q16_fid, x_stage2, feature_extractor, config['data'] == 'cifar10')
fid_stage3 = compute_fid_batch(x_q32_fid, x_stage3, feature_extractor, config['data'] == 'cifar10')

print(f'Stage 1 (8x8 blocky)   FID = {fid_stage1:.4f}')
print(f'Stage 2 (16x16 blocky) FID = {fid_stage2:.4f}')
print(f'Stage 3 (32x32 réel)   FID = {fid_stage3:.4f}')

pd.DataFrame({
    'stage': ['8x8', '16x16', '32x32'],
    'network': ['f', 'f16', 'f32'],
    'FID': [fid_stage1, fid_stage2, fid_stage3],
}).to_csv(OUT_DIR + 'fid_per_stage.csv', index=False)

# -------- FID intermédiaire (en fonction du pas Langevin) par étage --------
print('\n=== FID intermédiaire — stage 1 (8x8) ===')
steps1, fidcurve1 = fid_over_snapshots(snaps1, x_q8_fid)
print('\n=== FID intermédiaire — stage 2 (16x16) ===')
steps2, fidcurve2 = fid_over_snapshots(snaps2, x_q16_fid)
print('\n=== FID intermédiaire — stage 3 (32x32) ===')
steps3, fidcurve3 = fid_over_snapshots(snaps3, x_q32_fid)

pd.DataFrame({'step': steps1, 'fid': fidcurve1}).to_csv(OUT_DIR + 'fid_vs_step_stage1_8x8.csv', index=False)
pd.DataFrame({'step': steps2, 'fid': fidcurve2}).to_csv(OUT_DIR + 'fid_vs_step_stage2_16x16.csv', index=False)
pd.DataFrame({'step': steps3, 'fid': fidcurve3}).to_csv(OUT_DIR + 'fid_vs_step_stage3_32x32.csv', index=False)


# =========================
# PLOTS
# =========================

# -------- images générées par étage --------
plot_ims(OUT_DIR + 'gen_stage1_8x8.png', x_stage1[:N_IMAGES_PLOT])
plot_ims(OUT_DIR + 'gen_stage2_16x16.png', x_stage2[:N_IMAGES_PLOT])
plot_ims(OUT_DIR + 'gen_stage3_32x32.png', x_stage3[:N_IMAGES_PLOT])

# -------- images réelles de référence (mêmes résolutions) pour comparaison visuelle --------
plot_ims(OUT_DIR + 'real_stage1_8x8.png', x_q8_fid[:N_IMAGES_PLOT])
plot_ims(OUT_DIR + 'real_stage2_16x16.png', x_q16_fid[:N_IMAGES_PLOT])
plot_ims(OUT_DIR + 'real_stage3_32x32.png', x_q32_fid[:N_IMAGES_PLOT])

# -------- grille récapitulative : progression de la cascade pour les mêmes échantillons --------
n_show = min(8, N_IMAGES_PLOT)
fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
stage_imgs = [x_stage1, x_stage2, x_stage3]
stage_titles = ['Stage 1 (8x8)', 'Stage 2 (16x16)', 'Stage 3 (32x32)']
for row, (imgs, title) in enumerate(zip(stage_imgs, stage_titles)):
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
plt.suptitle('Progression de la cascade (mêmes échantillons à travers f -> f16 -> f32)')
plt.tight_layout()
plt.savefig(OUT_DIR + 'cascade_progression.png', dpi=150)
plt.close()

# -------- courbes gradient moyen par étage --------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, grads, title in zip(axes, [grads1, grads2, grads3], stage_titles):
    ax.plot(grads, color='steelblue')
    ax.set_xlabel('Langevin step')
    ax.set_ylabel('Mean gradient norm')
    ax.set_title(title)
plt.tight_layout()
plt.savefig(OUT_DIR + 'gradient_norm_per_stage.png', dpi=150)
plt.close()

# -------- courbe FID vs step, une couleur par étage --------
plt.figure(figsize=(10, 5))
plt.plot(steps1, fidcurve1, marker='o', markersize=3, label='Stage 1 (8x8) vs réel 8x8')
plt.plot(steps2, fidcurve2, marker='o', markersize=3, label='Stage 2 (16x16) vs réel 16x16')
plt.plot(steps3, fidcurve3, marker='o', markersize=3, label='Stage 3 (32x32) vs réel 32x32')
plt.xlabel('Langevin step (dans l\'étage)')
plt.ylabel('FID')
plt.title(f'FID vs pas Langevin par étage (batch={FID_BATCH_SIZE}, init stage1={STAGE1_INIT})')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + 'fid_vs_step_all_stages.png', dpi=150)
plt.close()

# -------- barplot FID final par résolution --------
plt.figure(figsize=(6, 4))
plt.bar(['8x8\n(f)', '16x16\n(f16)', '32x32\n(f32)'], [fid_stage1, fid_stage2, fid_stage3],
        color=['steelblue', 'darkorange', 'seagreen'])
plt.ylabel('FID final')
plt.title('FID final par résolution de la cascade')
plt.tight_layout()
plt.savefig(OUT_DIR + 'fid_final_per_stage.png', dpi=150)
plt.close()

print(f'\nTout a été sauvegardé dans : {OUT_DIR}')
print(f'  - gen_stage{{1,2,3}}_*.png / real_stage{{1,2,3}}_*.png : images générées/réelles par résolution')
print(f'  - cascade_progression.png : mêmes échantillons à travers les 3 étages')
print(f'  - fid_per_stage.csv, fid_final_per_stage.png : FID final par résolution')
print(f'  - fid_vs_step_stage*_*.csv, fid_vs_step_all_stages.png : FID en fonction du pas Langevin')
print(f'  - gradient_norm_per_stage.png : norme du gradient moyen par étage')