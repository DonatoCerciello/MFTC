"""Dataset loading, adaptive batch size and seeding."""
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
from scipy.io import arff
import torch
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_dataset(data_root, dataset):
    """Read a UEA dataset stored as one ARFF file per dimension.

    Expected files: <data_root>/<dataset>/<dataset>Dimension<d>_{TRAIN,TEST}.arff.
    Missing values and shorter series are zero-filled, then every series and channel is
    standardized independently. Labels are used only for evaluation and to set K.
    Returns ([x_train, x_test], [y_train, y_test], K) with x of shape [N, T, C].
    """
    root = Path(data_root) / dataset
    loaded, raw_labels = [], []
    for split in ['TRAIN', 'TEST']:
        paths = sorted(root.glob(f'{dataset}Dimension*_{split}.arff'),
                       key=lambda p: int(re.search(r'Dimension(\d+)', p.name).group(1)))
        if not paths:
            raise FileNotFoundError(f'No {dataset}Dimension*_{split}.arff files in {root}')
        channels = []
        for path in paths:
            table = pd.DataFrame(arff.loadarff(path)[0])
            channels.append(table.iloc[:, :-1].fillna(0).to_numpy(dtype=np.float32))
        loaded.append(torch.tensor(TimeSeriesScalerMeanVariance().fit_transform(
            np.stack(channels, axis=-1)), dtype=torch.float32))
        raw_labels.append(table.iloc[:, -1].to_numpy())
    classes = np.unique(raw_labels[0])
    ys = [torch.tensor(np.searchsorted(classes, y), dtype=torch.long) for y in raw_labels]
    return loaded, ys, len(classes)


def adaptive_batch_size(n_samples, n_clusters, base=16):
    """Batch size scaled with the number of clusters, keeping at least six batches per epoch
    (and at least four series per batch): min(N, max(base, 8K), max(N // 6, 4))."""
    return int(min(n_samples, max(base, 8 * n_clusters), max(4, n_samples // 6)))
