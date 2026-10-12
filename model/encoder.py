"""MFTC encoder: input projection with a learnable mask token, unidirectional Mamba blocks,
and the temporal pyramid readout that produces one vector per series."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

LEVELS = (1, 2, 4)


@dataclass(frozen=True)
class EncoderConfig:
    input_dims: int
    hidden_dims: int = 64
    output_dims: int = 64
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.2
    missing_rate: float = 0.2   # probability of replacing a projected timestamp with the mask token

    def __post_init__(self):
        if min(self.input_dims, self.hidden_dims, self.output_dims, self.n_layers,
               self.d_state, self.d_conv, self.expand) < 1:
            raise ValueError('Encoder dimensions must be positive')
        if not 0 <= self.missing_rate < 1 or not 0 <= self.dropout < 1:
            raise ValueError('missing_rate and dropout must be in [0, 1)')


class MambaBlock(nn.Module):
    """Pre-norm residual block with a Mamba-1 selective state-space mixer (forward in time)."""

    def __init__(self, config):
        super().__init__()
        from mamba_ssm.modules.mamba_simple import Mamba
        self.norm = nn.RMSNorm(config.hidden_dims)
        self.mixer = Mamba(d_model=config.hidden_dims, d_state=config.d_state,
                           d_conv=config.d_conv, expand=config.expand)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


def segments(length, parts):
    """Bounds [s, e) of `parts` consecutive, non-empty regions of equal duration in [0, length)."""
    bounds = []
    for i in range(parts):
        s = min(int(np.floor(i * length / parts)), length - 1)
        e = min(max(int(np.floor((i + 1) * length / parts)), s + 1), length)
        bounds.append((s, e))
    return bounds


def pyramid_pool(h):
    """[B, T, D] -> [B, 7 * D]: featurewise max over each region, blocks ordered 1 | 2 | 4."""
    blocks = [h[:, s:e].amax(1) for parts in LEVELS for s, e in segments(h.shape[1], parts)]
    return torch.cat(blocks, dim=-1)


class MFTCEncoder(nn.Module):
    blocks = sum(LEVELS)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dims, config.hidden_dims)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.out_norm = nn.LayerNorm(config.hidden_dims)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_dims, config.output_dims)
        # Learnable mask token, initialized from the input bias (no random draw).
        self.mask_token = nn.Parameter(self.input_proj.bias.detach().clone())

    def forward(self, x, corrupt=False):
        """x [B, T, C] -> timestamp representations H [B, T, output_dims].

        corrupt=True replaces each projected timestamp with the mask token with probability
        missing_rate (independently per series and timestamp)."""
        if x.ndim != 3 or x.shape[-1] != self.config.input_dims or x.shape[1] < 1:
            raise ValueError('Expected nonempty [B, T, C]')
        if not torch.isfinite(x).all():
            raise ValueError('Expected finite preprocessed inputs')
        h = self.input_proj(x)
        if corrupt:
            keep = torch.rand(x.shape[:2], device=x.device) >= self.config.missing_rate
            h = torch.where(keep.unsqueeze(-1), h, self.mask_token)
        for block in self.layers:
            h = block(h)
        return self.out_proj(self.dropout(self.out_norm(h)))

    def pool(self, x, corrupt=False):
        """Raw (uncentered, unnormalized) pyramid descriptor [B, 7 * output_dims]."""
        return pyramid_pool(self(x, corrupt=corrupt))
