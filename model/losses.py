"""Hierarchical contrastive loss on contextual crops (self-supervised term of MFTC).

hierarchical_contrastive_loss, instance_contrastive_loss and temporal_contrastive_loss are
adapted from TS2Vec (Yue et al., AAAI 2022, https://github.com/zhihanyue/ts2vec, MIT License),
with the temporal contrast applied at every level (TS2Vec temporal_unit = 0).
"""
import numpy as np
import torch
from torch.nn import functional as F


def hierarchical_contrastive_loss(z1, z2, alpha=0.5):
    """Mean over the levels of alpha * instance + (1 - alpha) * temporal contrast; each level
    halves the time axis by max pooling, until a single timestamp remains."""
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if 1 - alpha != 0:
            loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def contextual_crops(x):
    """Two overlapping crops with a random per-series offset (TS2Vec contextual consistency).

    The overlap length L is drawn from {2, ..., T}. Returns the crops and the slices of their
    outputs that cover the same timestamps.
    """
    b, t, _ = x.shape
    if t < 2:
        return x, x, slice(None), slice(None)
    length = int(np.random.randint(2, t + 1))
    left = int(np.random.randint(t - length + 1))
    right = left + length
    extended_left = int(np.random.randint(left + 1))
    extended_right = int(np.random.randint(right, t + 1))
    offsets = np.random.randint(-extended_left, t - extended_right + 1, size=b)

    def take(starts, size):
        index = torch.as_tensor(starts, device=x.device)[:, None]
        index = index + torch.arange(size, device=x.device)[None, :]
        return x[torch.arange(b, device=x.device)[:, None], index]

    a = take(offsets + extended_left, right - extended_left)
    c = take(offsets + left, extended_right - left)
    return a, c, slice(-length, None), slice(0, length)


def ssl_loss(encoder, x, temporal_weight=.5, max_train_length=4000):
    """L_hier: hierarchical instance + temporal contrast between two masked contextual crops."""
    if not 0 <= temporal_weight <= 1:
        raise ValueError('temporal_weight must be in [0,1]')
    if x.shape[1] > max_train_length:
        start = int(np.random.randint(x.shape[1] - max_train_length + 1))
        x = x[:, start:start + max_train_length]
    a, b, sa, sb = contextual_crops(x)
    za = encoder(a, corrupt=True)[:, sa]
    zb = encoder(b, corrupt=True)[:, sb]
    loss = hierarchical_contrastive_loss(za, zb, alpha=1 - temporal_weight)
    # Keeps every output in the graph when the loss is a constant (B = T = 1).
    return loss + 0 * (za.sum() + zb.sum())
