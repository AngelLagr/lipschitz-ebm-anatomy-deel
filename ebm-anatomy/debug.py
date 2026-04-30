from nets import *
import torch
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

# initialisation du réseau
f = VanillaNetOrthogonium()
f = f.to(device)

# résumé du modèle
summary(f, (1, 32, 32))

# -----------------------------
# test norme du gradient
# -----------------------------

batch_size = 128

x = torch.randn(batch_size, 1, 32, 32, device=device, requires_grad=True)

# petit entraînement pour que les gradients soient non nuls
optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)
for _ in range(1):
    optimizer.zero_grad()
    output = f(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()

y = f(x).sum()

grad = torch.autograd.grad(y, x)[0]

grad_norm = grad.view(batch_size, -1).norm(dim=1)

print("Gradient norm stats:")
print("mean :", grad_norm.mean().item())
print("std  :", grad_norm.std().item())
print("min  :", grad_norm.min().item())
print("max  :", grad_norm.max().item())