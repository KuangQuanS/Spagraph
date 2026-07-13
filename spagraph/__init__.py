"""Spagraph: Spatial transcriptomics deconvolution with VAE + GAT + cell communication."""

from .__version__ import __version__, __author__, __email__, __description__
from .training import (
    train_vae, run_deconv, run_deconv_auto_k, run_cellcom,
    run_cellcom_ensemble, Stage1Artifacts,
)

# High-level aliases
vae = train_vae
deconv = run_deconv
deconv_auto_k = run_deconv_auto_k
cellcom = run_cellcom
cellcom_ensemble = run_cellcom_ensemble

__all__ = [
    '__version__',
    'train_vae',
    'vae',
    'run_deconv',
    'deconv',
    'run_deconv_auto_k',
    'deconv_auto_k',
    'Stage1Artifacts',
    'run_cellcom',
    'cellcom',
    'run_cellcom_ensemble',
    'cellcom_ensemble',
]
