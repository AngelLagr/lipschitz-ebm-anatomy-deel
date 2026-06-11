import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.fft as fft
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as tr
from deel import torchlip

from utilfid import InceptionFeatures, compute_fid_batch, load_mnist_feature_extractor
from utils import plot_ims


# =========================
# SIMPLE CONFIG
# =========================
EXP_DIR = './out_data/Expe/cifar10_Langevin10_WARMUP_alpha0999_lossnorm_featureonlyonHKR/'
CHECKPOINTS = {
    'stable': EXP_DIR + 'checkpoints/last_checkpoint.pth',
    'exploding': EXP_DIR + 'checkpoints/net_150000.pth',
}
CSV_FILE = EXP_DIR + 'plots/eval_smartloss_cifar_metrics.csv'
PLOT_DIR = EXP_DIR + 'plots/eval_smartloss_cifar/'

LONGRUN_STEPS = 10
LONGRUN_INIT = 'checkpoint'  # 'fft' | 'checkpoint'
LONGRUN_BATCH_SIZE = 100
FFT_NOISE_BANDS = ('low', 'mid', 'high')
SEED = 42
HKR_THRESHOLD = 0.0
HKR_TAU = 0.3

CONFIG_FILE = None
for file in os.listdir(EXP_DIR + 'code/'):
    if file.endswith('.json'):
        CONFIG_FILE = os.path.join(EXP_DIR + 'code/', file)
        break
if CONFIG_FILE is None:
    raise FileNotFoundError(f'No JSON config file found in {EXP_DIR}')

with open(CONFIG_FILE) as file:
    config = json.load(file)

if config['data'] != 'cifar10':
    raise RuntimeError('This script is CIFAR-only.')

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

if config['epsilon'] > 0:
    temp = (config['epsilon'] ** 2) / 2
else:
    temp = 1.0

if config['data'] == 'mnist':
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)
elif config['data'] == 'cifar10':
    feature_extractor = InceptionFeatures().half().to(device)
else:
    feature_extractor = None


# =========================
# SMARTLOSS MODEL
# =========================
class Identity(nn.Module):
    def forward(self, x):
        return x


def conv3x3_lip(in_planes, out_planes, stride=1):
    return torchlip.SpectralConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True,
    )


class wide_basic_lip(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = Identity()
        self.lip_residual = torchlip.LipResidual()
        self.act = torchlip.GroupSort2()

        self.conv1 = torchlip.SpectralConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = Identity()
        self.conv2 = torchlip.SpectralConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                torchlip.SpectralConv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=True)
            )

    def forward(self, x):
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        out = (out + self.lip_residual(out, self.shortcut(x))) / 2
        return out


class DualHeadLipEBM(nn.Module):
    def __init__(self, depth=16, widen_factor=2, n_c=3, n_classes=10, lip_const=1.0, hkr_switch_threshold=0.0, hkr_tau=0.1):
        super().__init__()
        self.lip_const = float(lip_const)
        self.hkr_switch_threshold = float(hkr_switch_threshold)
        self.hkr_tau = float(hkr_tau)
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'depth must be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.activation = torchlip.GroupSort2()
        self.lipavgpool2d = torchlip.ScaledAvgPool2d(kernel_size=8)

        self.conv1 = conv3x3_lip(n_c, n_stages[0])
        self.layer1 = self._wide_layer(n_stages[1], n, stride=1)
        self.layer2 = self._wide_layer(n_stages[2], n, stride=2)
        self.layer3 = self._wide_layer(n_stages[3], n, stride=2)
        self.bn1 = Identity()

        self.hkr_head = self._make_head(n_stages[3], n_classes, lip=True)
        self.ebm_head = self._make_head(n_stages[3], n_classes, lip=False)

    def _make_head(self, in_dim, out_dim, lip):
        hidden_dim = in_dim * 2
        layers = []
        if lip:
            layers.append(torchlip.SpectralLinear(in_dim, hidden_dim))
            layers.append(torchlip.GroupSort2())
            layers.append(torchlip.SpectralLinear(hidden_dim, hidden_dim))
            layers.append(torchlip.GroupSort2())
            layers.append(torchlip.SpectralLinear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.05))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.05))
            layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(wide_basic_lip(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.lipavgpool2d(out)
        return out.view(out.size(0), -1)

    def energy_components(self, x):
        h = self.features(x)
        hkr_logits = self.hkr_head(h)
        ebm_logits = self.ebm_head(h)
        hkr_energy = self.lip_const * hkr_logits.logsumexp(1)
        ebm_energy = ebm_logits.logsumexp(1)
        return ebm_energy, hkr_energy

    def forward(self, x, head='theta'):
        ebm_energy, hkr_energy = self.energy_components(x)

        if head in ('lip', 'hkr'):
            return hkr_energy
        if head in ('nonlip', 'ebm'):
            return ebm_energy
        if head == 'theta':
            alpha = t.sigmoid((hkr_energy - self.hkr_switch_threshold) / self.hkr_tau)
            alpha = alpha.detach()
            return alpha * hkr_energy + (1.0 - alpha) * ebm_energy
        raise RuntimeError('Invalid head (use "theta", "hkr" or "ebm")')


def build_model():
    return DualHeadLipEBM(
        depth=config.get('depth', 16),
        widen_factor=config.get('widen_factor', 2),
        n_c=config['im_ch'],
        n_classes=config.get('n_classes', 10),
        lip_const=config.get('Lip_const', 1.0),
        hkr_switch_threshold=HKR_THRESHOLD,
        hkr_tau=HKR_TAU,
    ).to(device)


# 2. load_checkpoint retourne aussi le dict brut
def load_checkpoint(model, checkpoint_path):
    checkpoint = t.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'lip_const_record' in checkpoint and len(checkpoint['lip_const_record']) > 0:
            model.lip_const = float(checkpoint['lip_const_record'][-1])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, checkpoint  # <-- on expose le dict


# =========================
# DATA / NOISE
# =========================
def make_freq_mask(sz, band, device):
    y = fft.fftfreq(sz, device=device).reshape(-1, 1)
    x = fft.fftfreq(sz, device=device).reshape(1, -1)
    r2 = x ** 2 + y ** 2
    if band == 'low':
        return t.exp(-r2 / (2 * 0.04 ** 2))
    if band == 'mid':
        low = t.exp(-r2 / (2 * 0.04 ** 2))
        high = t.exp(-r2 / (2 * 0.15 ** 2))
        return high - low
    if band == 'high':
        return 1 - t.exp(-r2 / (2 * 0.15 ** 2))
    raise RuntimeError('Invalid band for frequency mask (use "low", "mid", or "high")')


def sample_freq_noise(batch_size, band='mixed'):
    if band == 'mixed':
        bands = [random.choice(FFT_NOISE_BANDS) for _ in range(batch_size)]
    elif band in FFT_NOISE_BANDS:
        bands = [band] * batch_size
    else:
        raise RuntimeError('Invalid band for frequency noise (use "low", "mid", "high", or "mixed")')

    noise = t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']], device=device)
    freq = fft.fft2(noise)
    filtered = t.empty_like(noise)

    for current_band in FFT_NOISE_BANDS:
        current_inds = [idx for idx, band_name in enumerate(bands) if band_name == current_band]
        if not current_inds:
            continue
        mask = make_freq_mask(config['im_sz'], current_band, device)[None, None, :, :]
        filtered[current_inds] = fft.ifft2(freq[current_inds] * mask).real

    return filtered / (filtered.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0], device=image_set.device)[:batch_size]
    return image_set[rand_inds]


def sample_q(q, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + config['data_epsilon'] * t.randn_like(x_q)


# 3. Nouvelle fonction (calquée sur eval_hkr_simple)
def sample_checkpoint_noise(checkpoint, batch_size):
    if not (isinstance(checkpoint, dict) and 's_t_0' in checkpoint):
        raise RuntimeError('Checkpoint does not contain s_t_0')
    buffer = checkpoint['s_t_0'].to(device)
    if buffer.shape[0] < batch_size:
        raise RuntimeError(
            f's_t_0 buffer too small: has {buffer.shape[0]} samples, needs {batch_size}'
        )
    inds = t.randperm(buffer.shape[0], device=device)[:batch_size]
    return buffer[inds]


def sample_init_noise(checkpoint, batch_size):
    if LONGRUN_INIT == 'fft':
        return sample_freq_noise(batch_size, band='mixed')
    if LONGRUN_INIT == 'checkpoint':
        return sample_checkpoint_noise(checkpoint, batch_size)
    raise RuntimeError(f'Invalid LONGRUN_INIT: {LONGRUN_INIT!r} (use "fft" or "checkpoint")')


def load_cifar_data():
    transform = tr.Compose([
        tr.Resize(config['im_sz']),
        tr.CenterCrop(config['im_sz']),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
    ])
    dataset = datasets.CIFAR10(root='./data/cifar10', transform=transform, download=True)
    return t.stack([x[0] for x in dataset]).to(device)


# =========================
# LONG RUN METRICS
# =========================
def hkr_accuracy(scores, target_is_positive):
    if target_is_positive:
        return (scores > 0).float().mean().item()
    return (scores < 0).float().mean().item()


def langevin_longrun_theta(model, x_init):
    x = t.autograd.Variable(x_init.clone(), requires_grad=True)

    hkr_value_means = []
    hkr_value_stds = []
    grad_norms = []
    sigma_means = []
    acc_means = []

    for step in range(LONGRUN_STEPS):
        ebm_energy, hkr_energy = model.energy_components(x)
        alpha = t.sigmoid((hkr_energy - model.hkr_switch_threshold) / model.hkr_tau)
        theta_energy = alpha.detach() * hkr_energy + (1.0 - alpha.detach()) * ebm_energy
        grad = t.autograd.grad(theta_energy.sum(), [x])[0]
        x = (x - grad + config['epsilon'] * t.randn_like(x)).detach().requires_grad_(True)

        hkr_values = model(x, head='hkr')
        hkr_values_det = hkr_values.detach()
        grad_norm = grad.view(grad.shape[0], -1).norm(dim=1)

        hkr_value_means.append(hkr_values_det.mean().item())
        hkr_value_stds.append(hkr_values_det.std(unbiased=False).item())
        grad_norms.append(grad_norm.mean().item())
        sigma_means.append(alpha.detach().mean().item())
        acc_means.append(hkr_accuracy(hkr_values_det, target_is_positive=True))

        if step == 0 or (step + 1) % 10 == 0 or (step + 1) == LONGRUN_STEPS:
            print(
                'Step {} / {}  hkr_value={:>12.6f}  grad={:>12.6f}  alpha={:>12.6f}  acc(hkr)={:>10.4f}'.format(
                    step + 1,
                    LONGRUN_STEPS,
                    hkr_value_means[-1],
                    grad_norms[-1],
                    sigma_means[-1],
                    acc_means[-1],
                ),
                end='\r',
            )

    return x.detach(), {
        'hkr_value_mean': np.array(hkr_value_means),
        'hkr_value_std': np.array(hkr_value_stds),
        'grad_mean': np.array(grad_norms),
        'sigma_mean': np.array(sigma_means),
        'acc_mean': np.array(acc_means),
    }


def langevin_longrun_hkr(model, x_init):
    x = t.autograd.Variable(x_init.clone(), requires_grad=True)

    hkr_value_means = []
    hkr_value_stds = []
    grad_norms = []
    acc_means = []

    for step in range(LONGRUN_STEPS):
        hkr_values = model(x, head='hkr')
        grad = t.autograd.grad(hkr_values.sum(), [x])[0]
        x = (x - grad + config['epsilon'] * t.randn_like(x)).detach().requires_grad_(True)

        hkr_values_det = hkr_values.detach()
        grad_norm = grad.view(grad.shape[0], -1).norm(dim=1)

        hkr_value_means.append(hkr_values_det.mean().item())
        hkr_value_stds.append(hkr_values_det.std(unbiased=False).item())
        grad_norms.append(grad_norm.mean().item())
        acc_means.append(hkr_accuracy(hkr_values_det, target_is_positive=True))

        if step == 0 or (step + 1) % 10 == 0 or (step + 1) == LONGRUN_STEPS:
            print(
                'Step {} / {}  hkr_value={:>12.6f}  grad={:>12.6f}  acc(hkr)={:>10.4f}'.format(
                    step + 1,
                    LONGRUN_STEPS,
                    hkr_value_means[-1],
                    grad_norms[-1],
                    acc_means[-1],
                ),
                end='\r',
            )

    return x.detach(), {
        'hkr_value_mean': np.array(hkr_value_means),
        'hkr_value_std': np.array(hkr_value_stds),
        'grad_mean': np.array(grad_norms),
        'acc_mean': np.array(acc_means),
    }


def save_curve_plot(path, values, ylabel, title):
    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.xlabel('Langevin step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_hist_plot(path, data_scores, noise_scores, refined_scores, title):
    plt.figure(figsize=(8, 4))
    plt.hist(data_scores, bins=50, alpha=0.5, label='data')
    plt.hist(noise_scores, bins=50, alpha=0.5, label='noise init')
    plt.hist(refined_scores, bins=50, alpha=0.5, label='noise longrun')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    t.manual_seed(SEED)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    data = load_cifar_data()

    all_rows = []

    for tag, checkpoint_path in CHECKPOINTS.items():
        if not os.path.exists(checkpoint_path):
            print(f'Skip {tag}: checkpoint not found at {checkpoint_path}')
            continue

        print(f'\n=== Evaluating {tag} ===')
        model, checkpoint = load_checkpoint(build_model(), checkpoint_path) 

        x_eval = sample_q(data, LONGRUN_BATCH_SIZE)
        x_noise_init = sample_init_noise(checkpoint, LONGRUN_BATCH_SIZE)    # remplace l'appel direct
        
        with t.no_grad():
            data_scores = model(x_eval, head='hkr').detach().cpu().numpy()
            noise_scores = model(x_noise_init, head='hkr').detach().cpu().numpy()

        data_acc = hkr_accuracy(t.from_numpy(data_scores), target_is_positive=False)
        noise_acc = hkr_accuracy(t.from_numpy(noise_scores), target_is_positive=True)

        x_theta_longrun, theta_curves = langevin_longrun_theta(model, x_noise_init)
        fid_theta_longrun = compute_fid_batch(
            x_eval,
            x_theta_longrun,
            feature_extractor,
            config['data'] == 'cifar10',
        )

        with t.no_grad():
            theta_refined_scores = model(x_theta_longrun, head='hkr').detach().cpu().numpy()

        theta_refined_acc = hkr_accuracy(t.from_numpy(theta_refined_scores), target_is_positive=True)

        x_hkr_longrun, hkr_curves = langevin_longrun_hkr(model, x_noise_init)
        with t.no_grad():
            hkr_refined_scores = model(x_hkr_longrun, head='hkr').detach().cpu().numpy()

        hkr_refined_acc = hkr_accuracy(t.from_numpy(hkr_refined_scores), target_is_positive=True)

        print('\n')
        print(
            f'{tag}: data_acc={data_acc:.4f}  noise_acc={noise_acc:.4f}  '
            f'theta_refined_acc={theta_refined_acc:.4f}  hkr_refined_acc={hkr_refined_acc:.4f}  '
            f'fid_theta_longrun={fid_theta_longrun:.6f}'
        )

        tag_dir = os.path.join(PLOT_DIR, tag)
        os.makedirs(tag_dir, exist_ok=True)

        save_curve_plot(os.path.join(tag_dir, 'theta_longrun_hkr_value_curve.png'), theta_curves['hkr_value_mean'], 'HKR value', f'{tag} - theta longrun HKR value')
        save_curve_plot(os.path.join(tag_dir, 'theta_longrun_sigma_curve.png'), theta_curves['sigma_mean'], 'alpha', f'{tag} - theta longrun alpha')
        save_curve_plot(os.path.join(tag_dir, 'theta_longrun_accuracy_curve.png'), theta_curves['acc_mean'], 'accuracy', f'{tag} - theta longrun HKR accuracy')
        save_hist_plot(os.path.join(tag_dir, 'theta_longrun_hkr_energy_hist.png'), data_scores, noise_scores, theta_refined_scores, f'{tag} - HKR values after theta longrun')

        # plot dataaccuracy, noiseaccuracy, and refinedaccuracy in one plot
        plt.figure(figsize=(8, 4))
        plt.plot(theta_curves['acc_mean'], label='theta longrun acc')
        plt.hlines(data_acc, 0, LONGRUN_STEPS, colors='g', linestyles='dashed', label='data acc')
        plt.hlines(noise_acc, 0, LONGRUN_STEPS, colors='r', linestyles='dashed', label='noise init acc')
        plt.hlines(theta_refined_acc, 0, LONGRUN_STEPS, colors='b', linestyles='dashed', label='theta longrun refined acc')
        plt.xlabel('Langevin step')
        plt.ylabel('HKR accuracy')
        plt.title(f'{tag} - HKR accuracy during theta longrun')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(tag_dir, 'HKR_accuracy_comparison.png'), dpi=150)
        plt.close()

        plot_ims(os.path.join(tag_dir, 'data_sample.png'), x_eval[:16])
        plot_ims(os.path.join(tag_dir, 'noise_init_sample.png'), x_noise_init[:16])
        plot_ims(os.path.join(tag_dir, 'theta_longrun_sample.png'), x_theta_longrun[:16])
        plot_ims(os.path.join(tag_dir, 'hkr_longrun_sample.png'), x_hkr_longrun[:16])

        theta_step_df = pd.DataFrame({
            'step': np.arange(1, LONGRUN_STEPS + 1),
            'run_type': 'theta_longrun',
            'hkr_value_mean': theta_curves['hkr_value_mean'],
            'hkr_value_std': theta_curves['hkr_value_std'],
            'grad_mean': theta_curves['grad_mean'],
            'sigma_mean': theta_curves['sigma_mean'],
            'acc_mean': theta_curves['acc_mean'],
            'checkpoint': tag,
            'checkpoint_path': checkpoint_path,
            'data_acc': data_acc,
            'noise_acc': noise_acc,
            'refined_acc': theta_refined_acc,
            'fid_longrun': fid_theta_longrun,
        })
        hkr_step_df = pd.DataFrame({
            'step': np.arange(1, LONGRUN_STEPS + 1),
            'run_type': 'hkr_longrun',
            'hkr_value_mean': hkr_curves['hkr_value_mean'],
            'hkr_value_std': hkr_curves['hkr_value_std'],
            'grad_mean': hkr_curves['grad_mean'],
            'acc_mean': hkr_curves['acc_mean'],
            'checkpoint': tag,
            'checkpoint_path': checkpoint_path,
            'data_acc': data_acc,
            'noise_acc': noise_acc,
            'refined_acc': hkr_refined_acc,
        })
        all_rows.extend([theta_step_df, hkr_step_df])

    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f'\nSaved metrics to {CSV_FILE}')
    else:
        print('\nNo checkpoint was evaluated.')


if __name__ == '__main__':
    main()