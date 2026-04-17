"""
Compute Wasserstein distance between MNIST/CIFAR-10 distribution and uniform noise distribution
using the POT (Python Optimal Transport) library.
"""

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as tr
import numpy as np
import ot
import argparse


DATASET_CONFIG = {
    'mnist':   {'channels': 1, 'image_size': 28},
    'cifar10': {'channels': 3, 'image_size': 32},
}


def load_dataset_samples(dataset='mnist', num_samples=10000):
    cfg = DATASET_CONFIG[dataset]
    image_size = cfg['image_size']
    channels   = cfg['channels']

    transform = tr.Compose([
        tr.Resize(image_size),
        tr.CenterCrop(image_size),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5 * t.ones(channels)), tuple(0.5 * t.ones(channels))),
    ])

    dataset_loaders = {
        'mnist':   lambda: datasets.MNIST(root='./data/mnist',   train=True, transform=transform, download=True),
        'cifar10': lambda: datasets.CIFAR10(root='./data/cifar10', train=True, transform=transform, download=True),
    }

    ds = dataset_loaders[dataset]()
    indices = np.random.choice(len(ds), size=num_samples, replace=False)
    samples = np.array([ds[i][0].numpy() for i in indices], dtype=np.float32)
    return samples


def generate_uniform_noise(num_samples, channels, image_size):
    return np.random.uniform(-1, 1, size=(num_samples, channels, image_size, image_size)).astype(np.float32)


def compute_wasserstein_distance(samples1, samples2, metric='euclidean'):
    flat1 = samples1.reshape(samples1.shape[0], -1)
    flat2 = samples2.reshape(samples2.shape[0], -1)

    M = ot.dist(flat1, flat2, metric=metric)
    a = np.ones(len(flat1)) / len(flat1)
    b = np.ones(len(flat2)) / len(flat2)

    return ot.emd2(a, b, M)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--metric',      default='euclidean')
    args = parser.parse_args()

    cfg        = DATASET_CONFIG[args.dataset]
    image_size = cfg['image_size']
    channels   = cfg['channels']

    print(f"Dataset : {args.dataset}  ({channels}ch, {image_size}x{image_size})")

    print("Loading dataset samples...")
    real_samples = load_dataset_samples(args.dataset, args.num_samples)
    print(f"  shape: {real_samples.shape}")

    print("Generating uniform noise...")
    noise_samples = generate_uniform_noise(args.num_samples, channels, image_size)
    print(f"  shape: {noise_samples.shape}")

    print("Computing Wasserstein distance...")
    dist = compute_wasserstein_distance(real_samples, noise_samples, metric=args.metric)

    print(f"\nWasserstein distance ({args.dataset} vs uniform noise): {dist:.6f}")
    print("="*60)
    print(f"Real  - min: {real_samples.min():.4f}  max: {real_samples.max():.4f}  mean: {real_samples.mean():.4f}")
    print(f"Noise - min: {noise_samples.min():.4f}  max: {noise_samples.max():.4f}  mean: {noise_samples.mean():.4f}")


if __name__ == '__main__':
    main()