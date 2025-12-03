"""Training module for SC-MAP-ST pipeline

This module provides the three-stage training pipeline:
- Stage 1: VAE training for SC-ST integration
- Stage 2: GAT-based spatial deconvolution
- Stage 3: Cell-cell communication analysis (placeholder)
"""

from .stage1 import train_integration
from .stage2 import deconvolve_spots
from .stage3 import analyze_cellchat

__all__ = [
    'train_integration',
    'deconvolve_spots',
    'analyze_cellchat'
]
