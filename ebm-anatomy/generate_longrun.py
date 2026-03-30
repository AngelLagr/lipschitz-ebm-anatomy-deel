import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as tr
import json
import os
import numpy as np
import random

from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip
from utils import plot_ims


##################################
# ## PARAMS (like train_data) ## #
##################################

EXP_DIR = './out_data/Expe/mnist_Lip0_Langevin100_Seed42/'
CONFIG_FILE = './config_locker/mnist_Lip0_Langevin100_Seed42.json'
CHECKPOINT_PATH = EXP_DIR + 'checkpoints/last_checkpoint.pth'
LONGRUN_STEPS = 300000
BATCH_SIZE = 1
SEED = 42
#######################
# ## INITIAL SETUP ## #
#######################

with open(CONFIG_FILE) as file:
    config = json.load(file)

t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
OUT_DIR = EXP_DIR + 'longrun/'
os.makedirs(OUT_DIR, exist_ok=True)


####################
# ## LOAD MODEL ## #
####################

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

if not os.path.exists(CHECKPOINT_PATH):
    raise RuntimeError('Checkpoint not found: {}'.format(CHECKPOINT_PATH))

checkpoint = t.load(CHECKPOINT_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    f.load_state_dict(checkpoint['model_state_dict'])
else:
    f.load_state_dict(checkpoint)
f.eval()


############################################
# ## INIT PERSISTENT FROM MNIST (always) ## #
############################################

transform = tr.Compose([
    tr.Resize(config['im_sz']),
    tr.CenterCrop(config['im_sz']),
    tr.ToTensor(),
    tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
])

mnist_data = datasets.MNIST(root='./data/mnist', transform=transform, download=True)

q = t.stack([x[0] for x in mnist_data]).to(device)

s_t_0 = 2 * t.rand_like(q) - 1


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

    x_s_t = x_s_t_0.clone()
    for _ in range(L):
        print(f'Long-run sampling step {_+1}/{L}', end='\r')
        x_s_t = x_s_t.detach().requires_grad_(True)
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t = x_s_t + (-f_prime + config['epsilon'] * t.randn_like(x_s_t))

    return x_s_t.detach()

rand_inds = t.randperm(s_t_0.shape[0])[0:BATCH_SIZE]
x_s_t = sample_s_t(f, s_t_0, config, device, LONGRUN_STEPS, 'gaussian', BATCH_SIZE)

#########################
# ## LONGRUN SAMPLING ## #
#########################

x_longrun = x_s_t.detach()
plot_ims(OUT_DIR + 'longrun.png', x_longrun)
print('Saved longrun in: {}'.format(OUT_DIR))
