"""
Compute Wasserstein distance between MNIST distribution and uniform noise distribution
using the POT (Python Optimal Transport) library.
"""

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as tr
import numpy as np
import ot


def load_mnist_samples(num_samples=10000, image_size=28, device='cpu'):
    """
    Load MNIST dataset samples.
    
    Args:
        num_samples: Number of samples to load from MNIST
        image_size: Size to resize images to
        device: Device to load samples to ('cpu' or 'cuda')
    
    Returns:
        MNIST samples as numpy array of shape (num_samples, channels, height, width)
    """
    transform = tr.Compose([
        tr.Resize(image_size),
        tr.CenterCrop(image_size),
        tr.ToTensor(),
        tr.Normalize((0.5,), (0.5,)),
    ])
    
    mnist = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
    
    # Sample random indices
    indices = np.random.choice(len(mnist), size=num_samples, replace=False)
    
    # Load samples
    samples = []
    for idx in indices:
        img, _ = mnist[idx]
        samples.append(img.numpy())
    
    samples = np.array(samples, dtype=np.float32)
    return samples


def generate_uniform_noise(num_samples=10000, image_size=28):
    """
    Generate uniform noise samples.
    
    Args:
        num_samples: Number of noise samples to generate
        image_size: Size of images (flattened dimension will be image_size^2)
    
    Returns:
        Uniform noise samples as numpy array of shape (num_samples, 1, image_size, image_size)
    """
    noise = np.random.uniform(-1, 1, size=(num_samples, 1, image_size, image_size)).astype(np.float32)
    return noise


def compute_wasserstein_distance(samples1, samples2, metric='euclidean', p=2):
    """
    Compute Wasserstein distance between two distributions using POT.
    
    Args:
        samples1: First distribution samples, shape (n_samples, n_features)
        samples2: Second distribution samples, shape (m_samples, n_features)
        metric: Distance metric to use ('euclidean', 'sqeuclidean', etc.)
        p: Order of the Wasserstein distance (default: 2 for Wasserstein-2)
    
    Returns:
        Wasserstein distance value
    """
    # Keep image tensors unflattened in storage, but vectorize internally for OT.
    # This avoids the huge broadcasted tensor that caused the memory error.
    flat_samples1 = samples1.reshape(samples1.shape[0], -1)
    flat_samples2 = samples2.reshape(samples2.shape[0], -1)

    if metric == 'euclidean':
        M = ot.dist(flat_samples1, flat_samples2, metric='euclidean')
    elif metric == 'sqeuclidean':
        M = ot.dist(flat_samples1, flat_samples2, metric='sqeuclidean')
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Compute uniform weights for each distribution
    a = np.ones(samples1.shape[0]) / samples1.shape[0]  # weight for samples1
    b = np.ones(samples2.shape[0]) / samples2.shape[0]  # weight for samples2
    
    # Compute Wasserstein distance
    wasserstein_dist = ot.emd2(a, b, M)
    
    return wasserstein_dist


def main():
    """Main execution function."""
    print("Loading MNIST samples...")
    num_samples = 10000  # Exact OT becomes impractical at 10000 samples
    mnist_samples = load_mnist_samples(num_samples=num_samples, image_size=28)
    print(f"MNIST samples shape: {mnist_samples.shape}")
    
    print("\nGenerating uniform noise samples...")
    noise_samples = generate_uniform_noise(num_samples=num_samples, image_size=28)
    print(f"Noise samples shape: {noise_samples.shape}")
    
    print("\nComputing Wasserstein distance...")
    wasserstein_dist = compute_wasserstein_distance(mnist_samples, noise_samples, metric='euclidean')
    
    print(f"\nWasserstein distance between MNIST and uniform noise: {wasserstein_dist:.6f}")
    
    # Additional analysis
    print("\n" + "="*60)
    print("Distribution Statistics:")
    print("="*60)
    print(f"MNIST - Min: {mnist_samples.min():.4f}, Max: {mnist_samples.max():.4f}, Mean: {mnist_samples.mean():.4f}")
    print(f"Noise - Min: {noise_samples.min():.4f}, Max: {noise_samples.max():.4f}, Mean: {noise_samples.mean():.4f}")


if __name__ == '__main__':
    main()
