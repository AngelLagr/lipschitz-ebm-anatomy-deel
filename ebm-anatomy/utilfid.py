import torch as t
import torch.nn as nn
import numpy as np
from scipy import linalg
import os
import torch
from torchvision.models import inception_v3

class SimpleFeatureExtractor(nn.Module):
    """Simple CNN feature extractor for FID calculation"""
    def __init__(self, device):
        super(SimpleFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 256)
        self.to(device)
        self.eval()
    
    def forward(self, x):
        if x.shape[1] != 1:
            x = x[:, :1, :, :]
        if x.shape[-1] != 28:
            x = nn.functional.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def load_mnist_feature_extractor(ckpt_path, device):
    """Load pretrained MNIST feature extractor from local checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            'Checkpoint FID introuvable: {}. Lance trainfid.py pour le créer.'.format(ckpt_path)
        )

    model = SimpleFeatureExtractor(device)
    state = t.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(pretrained=True, aux_logits=True)  # False = toujours un tensor
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model

    def forward(self, x):
        x = x.half()
        # x : float [0, 1]  shape (B, 3, 32, 32)
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        out = self.model(((x - mean) / std).half())   # (B, 2048)

        if isinstance(out, tuple):  # aux_logits
            out = out[0]

        return out


@torch.no_grad()
def get_features(x, model):
    # x : float [-1, 1]  shape (N, 3, 32, 32)
    x = (x.clamp(-1, 1) + 1) / 2            # → [0, 1]
    feats = []
    for batch in x.split(16):
        feats.append(model(batch))
    return torch.cat(feats).cpu().numpy()    # (N, 2048)



def calculate_fid(real_features, generated_features):
    """Calculate Fréchet Inception Distance"""
    if isinstance(real_features, t.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(generated_features, t.Tensor):
        generated_features = generated_features.cpu().numpy()
    
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(generated_features, axis=0)
    
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(generated_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_fid_batch(real_batch, generated_batch, feature_extractor, cifar=False):
    """Compute FID for a single batch"""
    with t.no_grad():
        if cifar :
            real_feat = get_features(real_batch,feature_extractor)
            gen_feat = get_features(generated_batch,feature_extractor)
        else :
            real_feat = feature_extractor(real_batch).cpu().numpy()
            gen_feat = feature_extractor(generated_batch).cpu().numpy()
    
    fid = calculate_fid(real_feat, gen_feat)
    return fid
