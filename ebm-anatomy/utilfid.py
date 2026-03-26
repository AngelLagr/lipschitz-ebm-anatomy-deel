import torch as t
import torch.nn as nn
import numpy as np
from scipy import linalg
import os


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


def compute_fid_batch(real_batch, generated_batch, feature_extractor):
    """Compute FID for a single batch"""
    with t.no_grad():
        real_feat = feature_extractor(real_batch).cpu().numpy()
        gen_feat = feature_extractor(generated_batch).cpu().numpy()
    
    fid = calculate_fid(real_feat, gen_feat)
    return fid
