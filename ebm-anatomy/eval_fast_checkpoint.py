import json
import torch as t
import torchvision.transforms as tr
import torchvision.datasets as datasets
import pandas as pd
import argparse
from utils import*
from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip, VanillaNetLLip2, VanillaNet2, Wide_ResNet, Wide_ResNet_LLip
from utilfid import *

# =========================
# PARAMETRES (EN HAUT)
# =========================
# parse the args and set up experiment directory


EXP_DIR = './out_data/Expe/cifar10_Langevin100_ResNET16_3e-4_95/'
CONFIG_FILE = './config_locker/cifar10_Langevin100Du2.json'
CHECKPOINT_FILE = EXP_DIR + 'checkpoints/last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
FID_BATCH_SIZE = 100
LONGRUN_STEPS = 3000
LONGRUN_INIT = 'uniform' 
SEED = 42
# -------- load config/device --------
with open(CONFIG_FILE) as file:
    config = json.load(file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

WConstrained_enabled = config.get('Wasserstein_LipConstrained', False)
lip_const = None
# temperature from training
if config['epsilon'] > 0:
    temp = 1* (config['epsilon'] ** 2) / 2
else:
    temp = 1


############################################
# ## INIT PERSISTENT FROM MNIST (always) ## #
############################################
def langevin_grad(f, x_s_t_0_l, config, temp_l, lip_const=None):
    if WConstrained_enabled:
        f.Lip_const = lip_const
        
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    actual_batch_size = x_s_t_0_l.shape[0]  
    grads = t.zeros(LONGRUN_STEPS, actual_batch_size)
    ens = t.zeros(LONGRUN_STEPS, actual_batch_size)

    # iterative langevin updates of MCMC samples
    for ell in range(LONGRUN_STEPS):
        en = f(x_s_t_l) 
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]

        if config['epsilon'] > 0:
            x_s_t_l.data += - grad + config['epsilon'] * t.randn_like(x_s_t_l)
            grads[ell] =  grad.view(grad.shape[0], -1).norm(dim=1).cpu()
            
            
        else:
            x_s_t_l.data += - grad
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        if ell == 0 or (ell + 1) % 10 == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')
    return x_s_t_l.detach(), ens, grads



def sample_q(q, data_epsilon, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + data_epsilon * t.randn_like(x_q)


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]

def sample_s_t(f, config, device, L, init_type, batch_size, lip_const = None):
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)

    results = []
    for i in range(0, batch_size, 1000):
        print("Step :",i)
        print("\n")
        chunk = x_s_t_0[i:i+1000]
        out = langevin_grad(f, chunk, config, temp, lip_const)[0]
        results.append(out.cpu())
        t.cuda.empty_cache()

    return t.cat(results, dim=0).to(device)



# -------- build model --------
net_bank = {'vanilla': VanillaNet,'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet,'vanillaHalfLip': VanillaNetHalfLip,'vanillaLip2': VanillaNetLLip2, 'vanilla2': VanillaNet2, 'wideresnet':Wide_ResNet, "wideresnetLip" : Wide_ResNet_LLip}
if config['net_type'] in ("vanillaLip", "vanillaLip2", "wideresnetLip"):
    f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
elif (config['net_type']=="wideresnet22"):
    f = net_bank["wideresnet"](n_c=config['im_ch'],depth = 22).to(device)
    current_lip_const=None
elif (config['net_type']=="wideresnetLip22"):
    f = net_bank["wideresnetLip"](n_c=config['im_ch'], Lip_const=config['Lip_const'],depth = 22).to(device)
else :
    f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
    current_lip_const=None
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
# Initialize FID feature extractor for MNIST
if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None

with t.no_grad():
    x_q_fid = sample_q(q, config['data_epsilon'], FID_BATCH_SIZE)



# -------- compute long-run FID from noise init --------
x_s_t_longrun = sample_s_t(
    f=f,
    config=config,
    device=device,
    L=LONGRUN_STEPS,
    init_type=LONGRUN_INIT,
    batch_size=FID_BATCH_SIZE,
    lip_const = lip_const
)

#save a litle part of the long-run image for visualisation
x_s_t_longrun_vis = x_s_t_longrun[:16]
plot_ims(EXP_DIR + 'longrun/' + 'longrun_sample.png', x_s_t_longrun_vis)
plot_ims(EXP_DIR + 'longrun/' + 'q_sample.png', x_q_fid[:16])


fid_longrun = compute_fid_batch(x_q_fid, x_s_t_longrun, feature_extractor,config['data'] == 'cifar10')
print(f'FID_longrun_from_noise(batch={FID_BATCH_SIZE}, L={LONGRUN_STEPS}, init={LONGRUN_INIT}) = {fid_longrun:.6f}')

# -------- read csv and compute slope a --------
df = pd.read_csv(CSV_FILE)

iters = df['iteration'].to_numpy()
fids = df['FID'].to_numpy()

#Calculate the 3 best FID without taking into account the NaN values
fids_no_nan = fids[~pd.isna(fids)]
iters_no_nan = iters[~pd.isna(fids)]
top_k = min(3, len(fids_no_nan))
best_idx = np.argsort(fids_no_nan)[:top_k]
best_fids = fids_no_nan[best_idx]
best_iters = iters_no_nan[best_idx]
best_fid = best_fids[0]
best_iter = best_iters[0]

#print the 3 best FID points on the curve
print(f'Top {top_k} FID points:')
for i in range(top_k):
    print(f'Iter={best_iters[i]:.0f}, FID={best_fids[i]:.6f}')

# take the first point and the point where FID <= 10
idxfirst = 0
idx10 = next(i for i, v in enumerate(fids) if v <= 100)

x1, y1 = float(iters[idxfirst]), float(fids[idxfirst])
x2, y2 = float(iters[idx10]), float(fids[idx10])

a = (y2 - y1) / (x2 - x1)
b = y1 - a * x1

print(f'First point: iter={x1:.0f}, fid={y1:.6f}')
print(f'Point FID<=10: iter={x2:.0f}, fid={y2:.6f}')
print(f'Droite: f(x) = a*x + b avec a={a:.8f}, b={b:.8f}')

