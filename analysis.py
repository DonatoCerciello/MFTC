"""Analysis of the runs produced by train.py (no model is executed).

For every run <results>/<Dataset>/seed<N>/ it computes the latent-space scores of the test
embeddings after pretraining and after fine-tuning and, unless --no-plots is given, it saves in
<run>/plots/ the t-SNE projections (ground-truth classes and predicted clusters) of both stages and
the loss curves. Then it writes the summary tables in <results>/analysis/:

  per_run.csv        every metric of every run
  clustering.csv     test F1, NMI, ARI, RI, FMI per dataset (mean and std over seeds)
  refinement.csv     pretraining vs fine-tuning: native assignment and common train-fit KMeans head
  latent_space.csv   silhouette and kNN accuracy of the test embeddings, pretraining vs fine-tuning

Example:
    python analysis.py --results results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from utils.analysis import latent_scores, plot_losses, plot_projection, tsne_2d
from utils.metrics import hungarian_alignment

METRICS = ['F1', 'NMI', 'ARI', 'RI', 'FMI']
STAGES = ['pretraining', 'finetuning']


def analyze_run(run, args):
    record = json.loads((run / 'metrics.json').read_text())
    k = json.loads((run / 'config.json').read_text())['n_clusters']
    row = dict(dataset=record['dataset'], seed=record['seed'])
    row.update({m: record['finetuning'][m] for m in METRICS + ['ACC']})
    for stage in STAGES:
        row[f'{stage}_ARI'] = record[stage]['ARI']
        row[f'{stage}_NMI'] = record[stage]['NMI']
        row[f'{stage}_kmeans_ARI'] = record[f'{stage}_kmeans']['ARI']
        data = np.load(run / f'{stage}_predictions.npz')
        scores = latent_scores(data['embeddings'], data['labels'], args.knn)
        row[f'{stage}_silhouette'] = scores['silhouette']
        row[f'{stage}_knn'] = scores['knn']
        if not args.no_plots:
            folder = run / 'plots'
            folder.mkdir(exist_ok=True)
            xy = tsne_2d(data['embeddings'])
            plot_projection(xy, data['labels'], folder / f'{stage}_tsne_classes.png', 'Class', args.dpi)
            aligned = hungarian_alignment(data['labels'], data['predictions'], k)
            plot_projection(xy, aligned, folder / f'{stage}_tsne_clusters.png', 'Cluster', args.dpi)
    if not args.no_plots:
        plot_losses(pd.read_csv(run / 'pretraining_history.csv'), pd.read_csv(run / 'finetuning_history.csv'),
                    run / 'plots' / 'losses.png', args.dpi)
    return row


def per_dataset(table, columns):
    """Mean over seeds per dataset, plus a final row with the mean over datasets."""
    means = table.groupby('dataset')[columns].mean()
    means.loc['Mean'] = means.mean()
    return means


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', type=Path, default=Path('results'), help='output folder of train.py')
    p.add_argument('--knn', type=int, default=5, help='neighbours of the kNN accuracy')
    p.add_argument('--no-plots', action='store_true', help='skip t-SNE and loss plots')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()

    runs = sorted(r for r in args.results.glob('*/seed*') if (r / 'metrics.json').exists())
    if not runs:
        raise SystemExit(f'No completed runs in {args.results}')
    rows = []
    with threadpool_limits(limits=args.threads):
        for run in runs:
            print(f'{run.parent.name} {run.name}', flush=True)
            rows.append(analyze_run(run, args))
    table = pd.DataFrame(rows).sort_values(['dataset', 'seed'])

    out = args.results / 'analysis'
    out.mkdir(exist_ok=True)
    table.to_csv(out / 'per_run.csv', index=False)
    clustering = table.groupby('dataset')[METRICS].agg(['mean', 'std'])
    clustering.loc['Mean'] = clustering.mean()
    clustering.to_csv(out / 'clustering.csv')
    refinement = per_dataset(table, [f'{s}_{m}' for m in ['ARI', 'NMI', 'kmeans_ARI'] for s in STAGES])
    refinement.to_csv(out / 'refinement.csv')
    latent = per_dataset(table, [f'{s}_{m}' for m in ['silhouette', 'knn'] for s in STAGES])
    latent.to_csv(out / 'latent_space.csv')

    pd.set_option('display.width', 200)
    print('\nTest clustering metrics (mean over seeds)\n', per_dataset(table, METRICS).round(4).to_string())
    print('\nPretraining vs fine-tuning\n', refinement.round(4).to_string())
    print('\nLatent space of the test embeddings\n', latent.round(4).to_string())
    print(f'\nTables written to {out}')


if __name__ == '__main__':
    main()
