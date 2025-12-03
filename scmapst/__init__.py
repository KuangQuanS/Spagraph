"""
scmapst: Single-cell Mapping to Spatial Transcriptomics

A deep learning framework for spatial transcriptomics deconvolution.
"""

from .__version__ import __version__, __author__, __email__, __description__

# Import main API function
from .training.stage2 import deconvolve_spots

# Main API - just one function
deconvolve = deconvolve_spots

__all__ = [
    '__version__',
    'deconvolve',
]
