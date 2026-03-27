import json
import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import pandas as pd

from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip
from utilfid import load_mnist_feature_extractor, compute_fid_batch

# =========================
# PARAMETRES (EN HAUT)
# =========================
EXP_DIR = './out_data/Expe/mnist_Lip100_Langevin10_Seed42/'
CONFIG_FILE = './config_locker/mnist_Lip100_Langevin10_Seed42.json'
CHECKPOINT_FILE = EXP_DIR + 'checkpoints/last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 5000
LONGRUN_STEPS = 1000
LONGRUN_INIT = 'uniform' 

def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]


def sample_q(q, data_epsilon, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + data_epsilon * t.randn_like(x_q)


def sample_s_t(f, s_t_0, config, device, L, init_type, batch_size):
    if init_type == 'persistent':
        rand_inds = t.randperm(s_t_0.shape[0])[:batch_size]
        x_s_t_0 = s_t_0[rand_inds]
    elif init_type == 'data':
        x_s_t_0 = sample_q(q, config['data_epsilon'], batch_size)
    elif init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)

    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    for _ in range(L):
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t.data += -f_prime + config['epsilon'] * t.randn_like(x_s_t)

    return x_s_t.detach()


# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# -------- build model --------
net_bank = {
    'vanilla': VanillaNet,
    'vanillaLip': VanillaNetLLip,
    'nonlocal': NonlocalNet,
    'vanillaHalfLip': VanillaNetHalfLip,
}
if config['net_type'] == 'vanillaLip':
    f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
else:
    f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)

# -------- load checkpoint --------
checkpoint = t.load(CHECKPOINT_FILE, map_location=device)
f.load_state_dict(checkpoint['model_state_dict'])
f.eval()
s_t_0 = checkpoint['s_t_0'].to(device)
seed = checkpoint.get('seed', 42)
t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
import random
import numpy as np
np.random.seed(config['seed'])
random.seed(config['seed'])

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
feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)

# -------- compute long-run FID from noise init --------
x_s_t_longrun = sample_s_t(
    f=f,
    s_t_0=s_t_0,
    config=config,
    device=device,
    L=LONGRUN_STEPS,
    init_type=LONGRUN_INIT,
    batch_size=FID_BATCH_SIZE,
)

fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor)
print(f'FID_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {fid_longrun:.6f}')

# -------- read csv and compute slope a --------
df = pd.read_csv(CSV_FILE)

iters = df['iteration'].to_numpy()
fids = df['FID'].to_numpy()

# take the first point and the point where FID <= 10
idxfirst = 0
idx10 = next(i for i, v in enumerate(fids) if v <= 10)

x1, y1 = float(iters[idxfirst]), float(fids[idxfirst])
x2, y2 = float(iters[idx10]), float(fids[idx10])

a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

print(f'First point: iter={x1:.0f}, fid={y1:.6f}')
print(f'Point FID<=10: iter={x2:.0f}, fid={y2:.6f}')
print(f'Droite: f(x) = a*x + b avec a={a:.8f}, b={b:.8f}')
