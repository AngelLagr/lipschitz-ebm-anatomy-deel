import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EXPERIMENTS = [
    #'./out_data/Expe/mnist_Lip10_Langevin10_Seed42sgd',
    #'./out_data/Expe/mnist_Lip0_Langevin100_Seed42Vannilla2',
    './out_data/Expe/mnist_Lip40_Langevin10_Seed42sgd',
    './out_data/Expe/mnist_Lip40_loaded_on10_Langevin10_Seed42sgd',
]


def resolve_csv_path(item):
    if item.endswith('.csv'):
        return item
    return os.path.join(item, 'longrun', 'fid_vs_longrun_steps.csv')


def load_curve(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV introuvable: {csv_path}')

    df = pd.read_csv(csv_path)
    expected_cols = {'longrun_steps', 'FID'}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f'Colonnes attendues {expected_cols} absentes dans {csv_path}. '
            f'Colonnes trouvées: {list(df.columns)}'
        )

    return df.sort_values('longrun_steps')


def build_default_labels(exp_items):
    labels = []
    for item in exp_items:
        norm_item = os.path.normpath(item)
        if norm_item.endswith('.csv'):
            norm_item = os.path.dirname(os.path.dirname(norm_item))
        labels.append(os.path.basename(norm_item) or norm_item)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare plusieurs courbes FID vs longrun sur un meme graphique (axe Y logarithmique).'
    )
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=DEFAULT_EXPERIMENTS,
        help=(
            'Liste des experiences (dossier expe ou chemin CSV). Si dossier: '
            '<expe>/longrun/fid_vs_longrun_steps.csv sera utilise.'
        ),
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Labels a afficher dans la legende (meme longueur que --experiments).',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./out_data/Expe/compare_fid_longrun_logy.png',
        help='Chemin du plot de sortie.',
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Comparaison FID vs Longrun Steps',
        help='Titre du graphique.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.experiments):
        raise ValueError('Le nombre de labels doit correspondre au nombre d\'experiences.')

    labels = args.labels if args.labels is not None else build_default_labels(args.experiments)

    plt.figure(figsize=(9, 5.5))

    for exp_item, label in zip(args.experiments, labels):
        csv_path = resolve_csv_path(exp_item)
        curve_df = load_curve(csv_path)

        plt.plot(
            curve_df['longrun_steps'],
            curve_df['FID'],
            marker='o',
            linewidth=2,
            label=label,
        )
        print(f'Charge: {csv_path} -> {len(curve_df)} points')

    plt.yscale('log')
    plt.xlabel('Longrun steps (L)')
    plt.ylabel('FID (log scale)')
    plt.title(args.title)
    plt.grid(True, which='both', linestyle='--', alpha=0.45)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    plt.close()

    print(f'Plot comparaison sauvegarde: {args.output}')


if __name__ == '__main__':
    main()