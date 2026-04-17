from nets import*
import torch
from torchsummary import summary

f=Wide_ResNet_EBM()
f=f.to('cuda')
summary(f, (3,32,32))