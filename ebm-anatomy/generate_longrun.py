import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as tr
import json
import os
import numpy as np
import random

from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip, VanillaNetLLip2, VanillaNet2, Wide_ResNet, Wide_ResNet_LLip
from utils import plot_ims


##################################
# ## PARAMS (like train_data) ## #
##################################


EXP_DIR = './out_data/Expe/cifar/ResNet16/cifar10_Langevin100_ResNET16_3e-4_95/'
CONFIG_FILE = './config_locker/cifar10_Langevin100Du2.json'
CHECKPOINT_PATH = EXP_DIR + 'checkpoints/last_checkpoint.pth'
CSV_FILE = EXP_DIR + 'plots/training_metrics.csv'
LONGRUN_STEPS = 1500
BATCH_SIZE = 9
SEED = 42
EPSILON = (5e-3)

#######################
# ## INITIAL SETUP ## #
#######################

with open(CONFIG_FILE) as file:
    config = json.load(file)

# temperature from training
#temp=1
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

if not os.path.exists(CHECKPOINT_PATH):
    raise RuntimeError('Checkpoint not found: {}'.format(CHECKPOINT_PATH))

checkpoint = t.load(CHECKPOINT_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    f.load_state_dict(checkpoint['model_state_dict'])
else:
    f.load_state_dict(checkpoint)


f.to(device)
f.eval()


############################################
# ## INIT PERSISTENT FROM MNIST (always) ## #
############################################
def langevin_grad(f, x_s_t_0_l, config, temp_l):
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    # sampling records
    grads = t.zeros(LONGRUN_STEPS, BATCH_SIZE)
    ens = t.zeros(LONGRUN_STEPS, BATCH_SIZE)

    # iterative langevin updates of MCMC samples
    for ell in range(LONGRUN_STEPS):
        en = f(x_s_t_l) 
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        x_s_t_l.data += - grad + EPSILON * t.randn_like(x_s_t_l)
        grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        
        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')
    return x_s_t_l.detach(), ens, grads


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]

def sample_s_t(f, config, device, L, init_type, batch_size):
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']])).to(device) - 1
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)

     # iterative langevin updates of MCMC samples
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1).to(device)  # variable r_s_t (Section 3.2) to record average gradient magnitude
    for ell in range(L):
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t.data += - f_prime + EPSILON * t.randn_like(x_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()
        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == L:
            print('Step {} of {}.   Ave. r_s_t={:>14.9f}'.
                  format(ell+1, L, r_s_t.item()/(ell+1)), end='\r')

    return x_s_t.detach()
        

    return x_s_t.detach()

x_s_t = sample_s_t(f, config, device, LONGRUN_STEPS, 'uniform', BATCH_SIZE)

#########################
# ## LONGRUN SAMPLING ## #
#########################

x_longrun = x_s_t.detach()
plot_ims(OUT_DIR + 'longrun.png', x_longrun)
print('Saved longrun in: {}'.format(OUT_DIR))
