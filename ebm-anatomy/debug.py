from nets import*
import torch
from torchsummary import summary

f=VanillaNetLLip2(n_c=1, n_f=32)
f=f.to('cuda')
summary(f, (1,32,32))