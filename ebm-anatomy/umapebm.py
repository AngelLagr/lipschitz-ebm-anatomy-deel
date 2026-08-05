"""
UMAP du latent space Lipschitz du TestModel
Extrait les features AVANT ebm_head (après avgpool + flatten + Lip_const scaling)
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as tr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import umap  # pip install umap-learn

from nets import TestModel

# ── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = './out_data/Expe/cifar10_lip_lang10_testhalf100lip/checkpoints/last_checkpoint.pth'
CONFIG_FILE     = './config_locker/cifar10_Langevin10_halfLipDu3FFT.json'
N_SAMPLES       = 2000  
BATCH_SIZE      = 256
OUTPUT_FIG      = './umap_latent.png'
# ────────────────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chargement config + modèle
with open(CONFIG_FILE) as f:
    config = json.load(f)

model = TestModel(n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ── HOOK pour capturer le latent AVANT ebm_head ─────────────────────────────
latents = []

def hook_fn(module, input, output):
    # `input` contient le vecteur qui entre dans ebm_head
    latents.append(input[0].detach().cpu())

hook = model.ebm_head.register_forward_hook(hook_fn)

# ── Chargement données ───────────────────────────────────────────────────────
CIFAR10_CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

transform = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    tr.Normalize(tuple(0.5 * torch.ones(config['im_ch'])),
                 tuple(0.5 * torch.ones(config['im_ch']))),
])

dataset = datasets.CIFAR10(root='./data/cifar10', train=False,
                           transform=transform, download=True)

# Sous-échantillonnage aléatoire
indices = torch.randperm(len(dataset))[:N_SAMPLES]
subset  = torch.utils.data.Subset(dataset, indices)
loader  = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

all_labels = []

print(f"Extraction des features sur {N_SAMPLES} images...")
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        model(imgs)  # le hook remplit latents[]
        all_labels.append(labels)

hook.remove()

# Concaténation
features = torch.cat(latents, dim=0).numpy()       # (N, D)
labels   = torch.cat(all_labels, dim=0).numpy()    # (N,)
print(f"Shape latent space : {features.shape}")    # ex: (2000, 128)

# ── UMAP ─────────────────────────────────────────────────────────────────────
print("UMAP en cours...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(features)        # (N, 2)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
colors = cm.get_cmap('tab10', 10)

for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
    mask = labels == cls_idx
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               s=5, alpha=0.6, color=colors(cls_idx), label=cls_name)

ax.legend(markerscale=3, loc='best', fontsize=9)
ax.set_title('UMAP — Latent space Lipschitz (avant ebm_head)', fontsize=13)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)
print(f"Figure sauvegardée : {OUTPUT_FIG}")
plt.show()