from nets import *
import torch
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

# initialisation du réseau
f = Wide_ResNet(depth=22, widen_factor=4)
f = f.to(device)

# résumé du modèle
summary(f, (3, 32, 32))
