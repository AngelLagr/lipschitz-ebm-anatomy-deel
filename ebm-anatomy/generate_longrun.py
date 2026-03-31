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
LONGRUN_STEPS = 100000
BATCH_SIZE = 9
SEED = 42
EPSILON = (7.5e-3)

#######################
# ## INITIAL SETUP ## #
#######################

with open(CONFIG_FILE) as file:
    config = json.load(file)

# temperature from training
temp = 1 * (EPSILON ** 2) / 2
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
        en = f(x_s_t_l) / temp_l
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        if EPSILON > 0:
            x_s_t_l.data += - ((EPSILON**2)/2) * grad + EPSILON * t.randn_like(x_s_t_l)
            grads[ell] = ((EPSILON**2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()
            #x_s_t_l.data += - grad + EPSILON * t.randn_like(x_s_t_l)
            #grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        else:
            x_s_t_l.data += - grad
            grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean()), end='\r')
    return x_s_t_l.detach(), ens, grads

# langevin equation with MH adjustment
def langevin_mh(f, x_s_t_0_l, config, temp_l):
    x_s_t_l = t.autograd.Variable(x_s_t_0_l.clone(), requires_grad=True)

    # sampling records
    ens = t.zeros(LONGRUN_STEPS, BATCH_SIZE)
    grads = t.zeros(LONGRUN_STEPS, BATCH_SIZE)
    accepts = t.zeros(LONGRUN_STEPS)

    # iterative langevin updates of MCMC samples
    for ell in range(LONGRUN_STEPS):
        # get energy and gradient of current states
        en = f(x_s_t_l) / temp_l
        ens[ell] = en.detach().cpu()
        grad = t.autograd.grad(en.sum(), [x_s_t_l])[0]
        grads[ell] = ((EPSILON ** 2)/2) * grad.view(grad.shape[0], -1).norm(dim=1).cpu()

        # get initial gaussian momenta
        p = t.randn_like(x_s_t_l)

        # get proposal states
        x_prop = x_s_t_l - ((EPSILON ** 2)/2) * grad + EPSILON * p
        # update momentum
        en_prop = f(x_prop) / temp_l
        grad_prop = t.autograd.grad(en_prop.sum(), [x_prop])[0]
        p_prop = p - (EPSILON / 2) * (grad + grad_prop)

        # joint energy of states and auxiliary momentum variables
        joint_en_orig = en + 0.5 * t.sum((p ** 2).view(x_s_t_l.shape[0], -1), 1)
        joint_en_prop = en_prop + 0.5 * t.sum((p_prop ** 2).view(x_s_t_l.shape[0], -1), 1)

        # accept or reject states_prop using MH acceptance ratio
        accepted_proposals = t.rand_like(en) < t.exp(joint_en_orig - joint_en_prop)

        # update only states with accepted proposals
        x_s_t_l.data[accepted_proposals] = x_prop.data[accepted_proposals]
        accepts[ell] = float(accepted_proposals.sum().cpu()) / float(BATCH_SIZE)

        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == LONGRUN_STEPS:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}   Accept Rate={:>14.9f}'.
                  format(ell+1, LONGRUN_STEPS, ens[ell].mean(), grads[ell].mean(), accepts[ell]), end='\r')

    return x_s_t_l.detach(), ens, grads, accepts


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]

def sample_s_t(f, config, device, L, init_type, batch_size):
    if init_type == 'uniform':
        x_s_t_0 = (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']])).to(device) - 1
    elif init_type == 'gaussian':
        x_s_t_0 = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)

    x_s_t = x_s_t_0[t.randperm(x_s_t_0.shape[0])[0:batch_size]]

    x_s_t = langevin_grad(f, x_s_t, config, temp)[0]
        

    return x_s_t.detach()

x_s_t = sample_s_t(f, config, device, LONGRUN_STEPS, 'uniform', BATCH_SIZE)

#########################
# ## LONGRUN SAMPLING ## #
#########################

x_longrun = x_s_t.detach()
plot_ims(OUT_DIR + 'longrun.png', x_longrun)
print('Saved longrun in: {}'.format(OUT_DIR))
