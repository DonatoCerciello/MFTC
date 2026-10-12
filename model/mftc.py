"""MFTC clustering model: student/teacher encoders, fuzzy prototypes and teacher targets.

Labels never enter these functions.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def blockwise_embed(pooled, mean, blocks):
    """Center with the training mean, L2-normalize every block, scale so that ||z|| = 1."""
    z = F.normalize((pooled - mean.to(pooled.device)).reshape(len(pooled), blocks, -1), dim=-1)
    return z.reshape(len(pooled), -1) / math.sqrt(blocks)


def fuzzy_log_membership(z, centers, m=1.5):
    if m <= 1:
        raise ValueError('The fuzzifier must exceed one')
    dist2 = (z[:, None] - centers[None]).square().sum(-1).clamp_min(1e-8)
    return torch.log_softmax(-dist2.log() / (m - 1), dim=-1)


def fuzzy_membership(z, centers, m=1.5):
    return fuzzy_log_membership(z, centers, m).exp()


def dec_sharpen(p):
    """Target distribution: p_ik^2 / F_k, renormalized (F_k = soft cluster frequency)."""
    p = p.square() / p.sum(0).clamp_min(1e-8)
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


def fuzzy_centers(z, membership, previous, m=1.5):
    """Fuzzy weighted centroids, normalized; empty clusters keep their previous center."""
    weights = membership.pow(m)
    mass = weights.sum(0)
    vector = weights.T @ z
    valid = (mass > 1e-8) & (vector.norm(dim=1) > 1e-8)
    result = previous.clone()
    result[valid] = F.normalize(vector[valid] / mass[valid, None], dim=1)
    return result


@torch.no_grad()
def pooled_embeddings(encoder, x, device, batch_size=64):
    """Clean pyramid descriptors of a whole collection (eval mode, no masking), on CPU."""
    training = encoder.training
    encoder.eval()
    try:
        return torch.cat([encoder.pool(batch.to(device)).cpu() for batch in x.split(batch_size)])
    finally:
        encoder.train(training)


@torch.no_grad()
def teacher_targets(teacher, x, centers, embed, device, batch_size, m=1.5):
    """Targets and agreement weights from two independently masked full-series views.

    target = DEC-sharpened mean membership of the two views;
    weight = 1 - JSD(q1 || q2) / log 2, clipped to [0, 1]."""
    teacher.eval()
    targets, weights = [], []
    for batch in x.split(batch_size):
        batch = batch.to(device)
        za = embed(teacher.pool(batch, corrupt=True))
        zb = embed(teacher.pool(batch, corrupt=True))
        qa = fuzzy_membership(za, centers, m)
        qb = fuzzy_membership(zb, centers, m)
        p = (qa + qb) / 2
        js = .5 * ((qa * (qa.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(1)
                   + (qb * (qb.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(1))
        targets.append(p.cpu())
        weights.append((1 - js / np.log(2)).clamp(0, 1).cpu())
    return dec_sharpen(torch.cat(targets)), torch.cat(weights)


class MFTC(nn.Module):
    """Student encoder, EMA teacher, training mean of the descriptors and K fuzzy prototypes.

    The mean and the prototypes are buffers fitted on the training collection only, so test
    series are assigned inductively."""

    def __init__(self, encoder, n_clusters, fuzzifier=1.5):
        super().__init__()
        self.encoder = encoder
        self.teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
        self.fuzzifier = fuzzifier
        self.blocks = encoder.blocks
        dims = self.blocks * encoder.config.output_dims
        self.register_buffer('centers', torch.zeros(n_clusters, dims))
        self.register_buffer('mean', torch.zeros(dims))

    def embed(self, pooled):
        return blockwise_embed(pooled, self.mean, self.blocks)

    @torch.no_grad()
    def update_teacher(self, decay=.99):
        """Exponential moving average of the student parameters."""
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.lerp_(student, 1 - decay)

    def prototype_loss(self, x, target, weights):
        """L_proto: agreement-weighted cross-entropy between teacher targets and the student
        memberships of two masked views."""
        target, weights = target.detach(), weights.detach()
        losses = []
        for _ in range(2):
            z = self.embed(self.encoder.pool(x, corrupt=True))
            logq = fuzzy_log_membership(z, self.centers, self.fuzzifier)
            losses.append(-(target * logq).sum(1))
        return (torch.stack(losses).mean(0) * weights).sum() / weights.sum().clamp_min(1e-8)

    @torch.no_grad()
    def predict(self, x, device, batch_size=64):
        """Cluster assignment and embeddings of unseen series."""
        z = self.embed(pooled_embeddings(self.encoder, x, device, batch_size))
        return fuzzy_membership(z, self.centers.cpu(), self.fuzzifier).argmax(1), z
