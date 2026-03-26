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
CHECKPOINT_PATH = EXP_DIR + 'checkpoints/ckpt_006200.pth'
LOG_FREQ = 100


#######################
# ## INITIAL SETUP ## #
#######################

with open(CONFIG_FILE) as file:
    config = json.load(file)

t.manual_seed(config['seed'])
if t.cuda.is_available():
    t.cuda.manual_seed_all(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

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

if isinstance(checkpoint, dict) and ('s_t_0' in checkpoint):
    s_t_0 = checkpoint['s_t_0'].to(device)
else:
    s_t_0 = q.clone()

rand_inds = t.randperm(s_t_0.shape[0])[0:config['batch_size']]
x_s_t = t.autograd.Variable(s_t_0[rand_inds].clone(), requires_grad=True)


#########################
# ## LONGRUN SAMPLING ## #
#########################

plot_ims(OUT_DIR + 'initial_states.png', x_s_t)

for ell in range(config['num_longrun_steps']):
    f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
    x_s_t.data += -f_prime + config['epsilon'] * t.randn_like(x_s_t)

    if ell == 0 or (ell + 1) % LOG_FREQ == 0 or (ell + 1) == config['num_longrun_steps']:
        print('Step {} / {}   mean|grad|={:>14.9f}'.format(
            ell + 1,
            config['num_longrun_steps'],
            f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()
        ))

x_longrun = x_s_t.detach()

plot_ims(OUT_DIR + 'longrun_last.png', x_longrun)
t.save(x_longrun.cpu(), OUT_DIR + 'longrun_last_tensor.pth')

print('Saved longrun in: {}'.format(OUT_DIR))
