"""MFTC model: encoder with mask token and temporal pyramid, losses, teacher-guided fuzzy clustering."""
from .encoder import EncoderConfig, MFTCEncoder
from .mftc import MFTC

__all__ = ['EncoderConfig', 'MFTCEncoder', 'MFTC']
