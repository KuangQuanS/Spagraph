"""Cell communication module for Spagraph.

This module provides cell-cell communication analysis based on
ligand-receptor interactions in spatial transcriptomics data.
"""

from .cellcom import main as run_main, parse_args
from .lr_scores import calculate_lr_scores

__all__ = [
    'run_main',
    'parse_args',
    'calculate_lr_scores',
]
