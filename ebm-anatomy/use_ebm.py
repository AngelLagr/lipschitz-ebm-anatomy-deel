import torch as t
import torchvision.transforms as tr
from PIL import Image
import json
import matplotlib.pyplot as plt
from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip
from utilfid import load_mnist_feature_extractor, compute_fid_batch

EXP_DIR = './out_data/Expe/mnist_Lip0_Langevin100_Seed42/'
CONFIG_FILE = './config_locker/mnist_Lip0_Langevin100_Seed42.json'
CHECKPOINT_PATH = EXP_DIR + 'checkpoints/last_checkpoint.pth'
LONGRUN_IMAGE_PATH = EXP_DIR + 'longrun/longrun.png'

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Load config and model
with open(CONFIG_FILE) as f:
    config = json.load(f)

net_bank = {'vanilla': VanillaNet, 'vanillaLip': VanillaNetLLip, 'nonlocal': NonlocalNet, 'vanillaHalfLip': VanillaNetHalfLip}
if config['net_type'] == 'vanillaLip':
    model = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=config['Lip_const']).to(device)
else:
    model = net_bank[config['net_type']](n_c=config['im_ch']).to(device)

checkpoint = t.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint)
model.eval()

# Load and preprocess image
mode = 'L' if config['im_ch'] == 1 else 'RGB'
transform = tr.Compose([tr.Resize(config['im_sz']), tr.CenterCrop(config['im_sz']), tr.ToTensor(), 
                        tr.Normalize(tuple(0.5*t.ones(config['im_ch'])), tuple(0.5*t.ones(config['im_ch'])))])
img_tensor = transform(Image.open(LONGRUN_IMAGE_PATH).convert(mode)).unsqueeze(0).to(device)

# Compute energy
with t.no_grad():
    energy = model(img_tensor)

print(f'Energy: {float(energy):.6f}')

