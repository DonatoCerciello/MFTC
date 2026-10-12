"""External clustering metrics on the test collection (labels used only here)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, f1_score, fowlkes_mallows_score,
                             normalized_mutual_info_score, rand_score)


def _contingency(y, pred, k):
    w = np.zeros((k, k))
    for p, t in zip(pred, y):
        w[p, t] += 1
    return w


def hungarian_alignment(y, pred, k):
    """Map every cluster to a class with the Hungarian algorithm."""
    w = _contingency(y, pred, k)
    rows, cols = linear_sum_assignment(w.max() - w)
    mapping = dict(zip(rows, cols))
    return np.array([mapping.get(p, p) for p in pred])


def clustering_metrics(y, pred, k):
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    w = _contingency(y, pred, k)
    rows, cols = linear_sum_assignment(w.max() - w)
    return dict(ARI=float(adjusted_rand_score(y, pred)),
                RI=float(rand_score(y, pred)),
                NMI=float(normalized_mutual_info_score(y, pred, average_method='geometric')),
                F1=float(f1_score(y, hungarian_alignment(y, pred, k), average='macro')),
                ACC=float(w[rows, cols].sum() / len(y)),
                FMI=float(fowlkes_mallows_score(y, pred)),
                occupied_clusters=int(len(np.unique(pred))))
