import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as tr

from nets import VanillaNet, NonlocalNet, VanillaNetHalfLip, VanillaNetLLip, VanillaNetLLip2, VanillaNet2
from utilfid import load_mnist_feature_extractor, compute_fid_batch


def sample_image_set(image_set, batch_size):
    rand_inds = t.randperm(image_set.shape[0])[:batch_size]
    return image_set[rand_inds]


def sample_q(q, data_epsilon, batch_size):
    x_q = sample_image_set(q, batch_size=batch_size)
    return x_q + data_epsilon * t.randn_like(x_q)


def build_longrun_step_list(max_steps):
    fixed_steps = [1, 5, 10, 100, 200, 300]
    hundred_steps = list(range(400, max_steps + 1, 100))
    out = sorted(set([s for s in fixed_steps + hundred_steps if s <= max_steps]))
    if max_steps not in out:
        out.append(max_steps)
    return sorted(out)


def init_state(config, batch_size, init_type, device):
    if init_type == 'uniform':
        return (2 * t.rand([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]) - 1).to(device)
    if init_type == 'gaussian':
        return t.randn([batch_size, config['im_ch'], config['im_sz'], config['im_sz']]).to(device)
    raise ValueError(f'Unknown init_type: {init_type}')


def compute_fid_curve(f, config, temp_l, init_type, batch_size, step_list, x_q_fid, feature_extractor, device):
    x_s_t = init_state(config, batch_size, init_type, device)
    x_s_t = t.autograd.Variable(x_s_t.clone(), requires_grad=True)

    target_steps = set(step_list)
    max_steps = max(step_list)
    curve = []

    for ell in range(max_steps):
        en = f(x_s_t) / temp_l
        grad = t.autograd.grad(en.sum(), [x_s_t])[0]

        if config['epsilon'] > 0:
            x_s_t.data += -((config['epsilon'] ** 2) / 2) * grad + config['epsilon'] * t.randn_like(x_s_t)
        else:
            x_s_t.data += -grad

        current_step = ell + 1
        if current_step in target_steps:
            fid_value = compute_fid_batch(x_q_fid, x_s_t.detach(), feature_extractor)
            curve.append((current_step, fid_value))
            print(f'L={current_step:>4d} -> FID={fid_value:.6f}')

    return curve, x_s_t.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LipConstant', type=int, default = 40)
    parser.add_argument('--LangevinSteps', type=int, default = 10)
    parser.add_argument('--LongrunSteps', type=int, default=1000)
    parser.add_argument('--FIDBatchSize', type=int, default=3000)
    parser.add_argument('--InitType', type=str, default='uniform', choices=['uniform', 'gaussian'])
    parser.add_argument('--Seed', type=int, default=42)
    parser.add_argument('--CheckpointName', type=str, default='net_017000.pth')
    args = parser.parse_args()

    exp_dir = './out_data/Expe/mnist_Lip{}_Langevin{}_Seed42sgd/'.format(args.LipConstant, args.LangevinSteps)
    config_file = './config_locker/mnist_Lip{}_Langevin{}_Seed42.json'.format(args.LipConstant, args.LangevinSteps)
    checkpoint_file = exp_dir + 'checkpoints/' + args.CheckpointName
    longrun_dir = exp_dir + 'longrun/'
    os.makedirs(longrun_dir, exist_ok=True)

    with open(config_file) as file:
        config = json.load(file)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if config['epsilon'] > 0:
        temp = (config['epsilon'] ** 2) / 2
    else:
        temp = 1

    t.manual_seed(args.Seed)
    np.random.seed(args.Seed)
    random.seed(args.Seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.Seed)

    net_bank = {
        'vanilla': VanillaNet,
        'vanillaLip': VanillaNetLLip,
        'nonlocal': NonlocalNet,
        'vanillaHalfLip': VanillaNetHalfLip,
        'vanillaLip2': VanillaNetLLip2,
        'vanilla2': VanillaNet2,
    }

    if config['net_type'] in ("vanillaLip", "vanillaLip2"):
        f = net_bank[config['net_type']](n_c=config['im_ch'], Lip_const=10).to(device)
    else:
        f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)

    checkpoint = t.load(checkpoint_file, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        f.load_state_dict(checkpoint['model_state_dict'])
    else:
        f.load_state_dict(checkpoint)
    f.eval()

    data = {
        'cifar10': lambda path, func: datasets.CIFAR10(root=path, transform=func, download=True),
        'mnist': lambda path, func: datasets.MNIST(root=path, transform=func, download=True),
        'flowers': lambda path, func: datasets.ImageFolder(root=path, transform=func),
    }

    transform = tr.Compose([
        tr.Resize(config['im_sz']),
        tr.CenterCrop(config['im_sz']),
        tr.ToTensor(),
        tr.Normalize(tuple(0.5 * t.ones(config['im_ch'])), tuple(0.5 * t.ones(config['im_ch']))),
    ])

    q = t.stack([x[0] for x in data[config['data']]('./data/' + config['data'], transform)]).to(device)
    feature_extractor = load_mnist_feature_extractor('./fidcheckpoints/mnist_feature_extractor.pth', device)

    with t.no_grad():
        x_q_fid = sample_q(q, config['data_epsilon'], args.FIDBatchSize)

    step_list = build_longrun_step_list(args.LongrunSteps)
    print('Longrun steps used for curve:', step_list)

    curve_points, _ = compute_fid_curve(
        f=f,
        config=config,
        temp_l=temp,
        init_type=args.InitType,
        batch_size=args.FIDBatchSize,
        step_list=step_list,
        x_q_fid=x_q_fid,
        feature_extractor=feature_extractor,
        device=device,
    )

    curve_df = pd.DataFrame(curve_points, columns=['longrun_steps', 'FID'])
    curve_csv_path = longrun_dir + 'fid_vs_longrun_steps.csv'
    curve_png_path = longrun_dir + 'fid_vs_longrun_steps.png'

    curve_df.to_csv(curve_csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(curve_df['longrun_steps'], curve_df['FID'], marker='o', linewidth=2)
    plt.xlabel('Longrun steps (L)')
    plt.ylabel('FID')
    plt.title('FID vs Longrun steps')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(curve_png_path, dpi=150)
    plt.close()

    print(f'CSV saved to: {curve_csv_path}')
    print(f'Plot saved to: {curve_png_path}')


if __name__ == '__main__':
    main()
