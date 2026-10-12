"""Train and evaluate MFTC on UEA multivariate datasets.

For every dataset and seed:
  1. self-supervised pretraining with the hierarchical contrastive loss L_hier;
  2. initialization: teacher <- student, training mean of the pyramid descriptors,
     K prototypes from KMeans on the training embeddings;
  3. fuzzy prototype-based fine-tuning, L_fine = L_hier + lambda_p(e) * L_proto, with an EMA teacher;
  4. inductive evaluation on the test collection (mean and prototypes fitted on training data only).

Example:
    python train.py --data-root data/Multivariate_arff --datasets BasicMotions --seeds 1 2 3
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset

from utils.data import adaptive_batch_size, read_dataset, seed_all
from model.encoder import EncoderConfig, MFTCEncoder
from model.losses import ssl_loss
from utils.metrics import clustering_metrics
from model.mftc import (MFTC, blockwise_embed, fuzzy_centers, fuzzy_membership, pooled_embeddings,
                        teacher_targets)

PAPER_DATASETS = ['ArticularyWordRecognition', 'BasicMotions', 'CharacterTrajectories', 'Cricket', 'ERing',
                  'Epilepsy', 'Handwriting', 'JapaneseVowels', 'Libras', 'LSST', 'NATOPS', 'PenDigits',
                  'RacketSports', 'SpokenArabicDigits', 'UWaveGestureLibrary']


def train_stage(model, x, loader, args, stage):
    """One training stage. Pretraining: L_hier with a plateau scheduler.
    Fine-tuning: prototypes and targets refreshed by the teacher at the start of every epoch."""
    fine = stage == 'finetuning'
    epochs = args.finetune_epochs if fine else args.pretrain_epochs
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.fine_lr if fine else args.lr,
                                  weight_decay=.01)
    scheduler = None if fine else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.2, patience=12)
    history = []
    for epoch in range(epochs):
        began = time.time()
        if fine:
            with torch.no_grad():
                raw = pooled_embeddings(model.teacher, x, args.device, args.batch_size).to(args.device)
                model.mean.copy_(raw.mean(0))
                z = model.embed(raw)
                memberships = fuzzy_membership(z, model.centers, args.fuzzifier)
                new = fuzzy_centers(z, memberships, model.centers, args.fuzzifier)
                model.centers.copy_(torch.nn.functional.normalize(
                    args.center_momentum * model.centers + (1 - args.center_momentum) * new, dim=1))
            targets, weights = teacher_targets(model.teacher, x, model.centers, model.embed, args.device,
                                               args.batch_size, args.fuzzifier)
        model.encoder.train()
        coefficient = args.prototype_weight * min(1., (epoch + 1) / max(1, args.prototype_warmup)) if fine else 0.
        total = ssl_total = proto_total = 0.
        seen = 0
        for batch, index in loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            ssl = ssl_loss(model.encoder, batch, args.temporal_weight, args.max_train_length)
            proto = ssl.new_zeros(())
            if fine:
                proto = model.prototype_loss(batch, targets[index].to(args.device), weights[index].to(args.device))
            loss = ssl + coefficient * proto
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {stage} loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if fine:
                model.update_teacher(args.teacher_decay)
            n = len(batch)
            seen += n
            total += float(loss.detach()) * n
            ssl_total += float(ssl.detach()) * n
            proto_total += float(proto.detach()) * n
        if scheduler is not None:
            scheduler.step(total / seen)
        history.append(dict(epoch=epoch + 1, total_loss=total / seen, ssl_loss=ssl_total / seen,
                            prototype_loss=proto_total / seen, prototype_weight=coefficient,
                            lr=optimizer.param_groups[0]['lr'], seconds=time.time() - began))
        if (epoch + 1) % args.log_every == 0 or epoch + 1 == epochs:
            print(f'  {stage} {epoch + 1}/{epochs} loss={total / seen:.4f}', flush=True)
    return history


@torch.no_grad()
def evaluate(model, xs, ys, args, k, seed, run, stage):
    """Native MFTC assignment (fuzzy prototypes) and a common train-fit KMeans head."""
    pred, z = model.predict(xs[1], args.device, args.batch_size)
    np.savez_compressed(run / f'{stage}_predictions.npz', labels=ys[1].numpy(), predictions=pred.numpy(),
                        embeddings=z.numpy())
    native = clustering_metrics(ys[1].numpy(), pred.numpy(), k)
    train, test = (pooled_embeddings(model.encoder, x, args.device, args.batch_size) for x in xs)
    mean = train.mean(0)
    ztr, zte = (blockwise_embed(v, mean, model.blocks).numpy() for v in (train, test))
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(ztr).predict(zte)
    return native, clustering_metrics(ys[1].numpy(), km, k)


def run_one(dataset, seed, args):
    run = args.output_dir / dataset / f'seed{seed}'
    if (run / 'metrics.json').exists() and not args.overwrite:
        print(f'{dataset} seed={seed}: already done, skipping (use --overwrite)', flush=True)
        return json.loads((run / 'metrics.json').read_text())
    run.mkdir(parents=True, exist_ok=True)
    xs, ys, k = read_dataset(args.data_root, dataset)
    args.batch_size = adaptive_batch_size(len(xs[0]), k, args.base_batch_size)
    print(f'{dataset} seed={seed}: train {tuple(xs[0].shape)}, test {tuple(xs[1].shape)}, K={k}, '
          f'batch={args.batch_size}', flush=True)

    seed_all(seed)
    config = EncoderConfig(xs[0].shape[2], hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                           n_layers=args.layers, missing_rate=args.missing_rate)
    model = MFTC(MFTCEncoder(config), k, fuzzifier=args.fuzzifier).to(args.device)
    settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    (run / 'config.json').write_text(json.dumps(dict(dataset=dataset, seed=seed, n_clusters=k,
                                                     encoder=asdict(config), args=settings), indent=2))
    loader = DataLoader(TensorDataset(xs[0], torch.arange(len(xs[0]))), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    started = time.time()

    pre_history = train_stage(model, xs[0], loader, args, 'pretraining')

    # Initialization of the teacher, the training mean and the prototypes.
    model.teacher.load_state_dict(model.encoder.state_dict(), strict=True)
    raw = pooled_embeddings(model.encoder, xs[0], args.device, args.batch_size)
    model.mean.copy_(raw.mean(0).to(args.device))
    centers = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed).fit(
        model.embed(raw).numpy()).cluster_centers_
    model.centers.copy_(torch.nn.functional.normalize(torch.as_tensor(centers, device=args.device), dim=1))
    seed_all(seed + 20000)
    pre_native, pre_kmeans = evaluate(model, xs, ys, args, k, seed, run, 'pretraining')

    fine_history = train_stage(model, xs[0], loader, args, 'finetuning')
    native, kmeans = evaluate(model, xs, ys, args, k, seed, run, 'finetuning')

    torch.save(dict(model=model.state_dict(), encoder_config=asdict(config), n_clusters=k,
                    fuzzifier=args.fuzzifier), run / 'final.pt')
    pd.DataFrame(pre_history).to_csv(run / 'pretraining_history.csv', index=False)
    pd.DataFrame(fine_history).to_csv(run / 'finetuning_history.csv', index=False)
    record = dict(dataset=dataset, seed=seed, finetuning=native, finetuning_kmeans=kmeans,
                  pretraining=pre_native, pretraining_kmeans=pre_kmeans, seconds=time.time() - started)
    (run / 'metrics.json').write_text(json.dumps(record, indent=2))
    print(f'{dataset} seed={seed}: ARI={native["ARI"]:.4f} NMI={native["NMI"]:.4f} '
          f'(after pretraining ARI={pre_native["ARI"]:.4f})', flush=True)
    del model
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True,
                   help='folder with one subfolder per dataset, containing <Dataset>Dimension<d>_{TRAIN,TEST}.arff')
    p.add_argument('--datasets', nargs='+', default=PAPER_DATASETS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--threads', type=int, default=2, help='CPU threads for torch and KMeans')
    g = p.add_argument_group('model and training (defaults = paper configuration)')
    g.add_argument('--pretrain-epochs', type=int, default=100)
    g.add_argument('--finetune-epochs', type=int, default=100)
    g.add_argument('--hidden-dims', type=int, default=64)
    g.add_argument('--output-dims', type=int, default=64)
    g.add_argument('--layers', type=int, default=4)
    g.add_argument('--missing-rate', type=float, default=.2, help='mask-token probability (keep probability 0.8)')
    g.add_argument('--base-batch-size', type=int, default=16)
    g.add_argument('--lr', type=float, default=1e-3, help='pretraining learning rate')
    g.add_argument('--fine-lr', type=float, default=1e-4, help='fine-tuning learning rate')
    g.add_argument('--temporal-weight', type=float, default=.5, help='weight of temporal vs instance contrast')
    g.add_argument('--max-train-length', type=int, default=4000,
                   help='series longer than this are randomly cropped during training (memory bound)')
    g.add_argument('--fuzzifier', type=float, default=1.5)
    g.add_argument('--prototype-weight', type=float, default=1.)
    g.add_argument('--prototype-warmup', type=int, default=10, help='epochs of linear warm-up of L_proto')
    g.add_argument('--teacher-decay', type=float, default=.99, help='EMA decay of the teacher')
    g.add_argument('--center-momentum', type=float, default=.9)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.device)
    records = []
    with threadpool_limits(limits=args.threads):
        for dataset in args.datasets:
            for seed in args.seeds:
                records.append(run_one(dataset, seed, args))
    rows = [dict(dataset=r['dataset'], seed=r['seed'], **{m: r['finetuning'][m] for m in
                                                          ['ARI', 'NMI', 'RI', 'FMI', 'F1', 'ACC']})
            for r in records]
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / 'results.csv', index=False)
    summary = table.groupby('dataset')[['ARI', 'NMI', 'RI', 'FMI', 'F1']].agg(['mean', 'std'])
    summary.to_csv(args.output_dir / 'summary.csv')
    print(table.groupby('dataset')[['ARI', 'NMI', 'RI', 'FMI', 'F1']].mean().round(4).to_string())


if __name__ == '__main__':
    main()
