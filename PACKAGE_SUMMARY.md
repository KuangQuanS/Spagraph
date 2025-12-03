# SC-MAP-ST Package Structure

## Overview

The SC-MAP-ST codebase has been packaged as `scmapst` - a Python package with clean API interface for spatial transcriptomics cell type deconvolution.

## Package Structure

```
/mnt/d/ST_Graduation_Project/
├── scmapst/                          # Main package directory
│   ├── __init__.py                   # Package entry point (exports train, deconvolve, analyze_cellchat)
│   ├── __version__.py                # Version: 0.1.0
│   ├── cli.py                        # Command-line interface
│   ├── training/                     # Training pipeline modules
│   │   ├── __init__.py
│   │   ├── stage1.py                 # API wrapper for Stage 1 (VAE training)
│   │   ├── stage2.py                 # API wrapper for Stage 2 (GAT deconvolution)
│   │   └── stage3.py                 # Placeholder for Stage 3 (CellChat)
│   ├── models/                       # Model definitions (placeholder)
│   │   └── __init__.py
│   ├── preprocessing/                # Data preprocessing (placeholder)
│   │   └── __init__.py
│   └── utils/                        # Utility functions (placeholder)
│       └── __init__.py
├── SC_MAP_ST/                        # Original implementation (used internally)
│   ├── stage1.py                     # Original Stage 1 implementation
│   ├── stage2.py                     # Original Stage 2 implementation
│   ├── deconv_model.py               # Model definitions
│   └── stage1_utils.py               # Utility functions
├── setup.py                          # Package installation configuration
├── pyproject.toml                    # Modern Python package configuration
├── MANIFEST.in                       # Include/exclude rules for packaging
├── README.md                         # User documentation
├── INSTALL.md                        # Installation and usage guide
├── requirements.txt                  # Python dependencies
├── example_usage.py                  # Example script
├── test_installation.py              # Installation test script
└── setup.sh                          # Quick setup script

```

## Key Features

### 1. Clean API Interface

**Python API:**
```python
import scmapst

# Stage 1: Train VAE
results = scmapst.train(
    sc_file="data/sc.h5ad",
    st_file="data/st.h5ad",
    output_dir="output/stage1/",
    n_epochs=150
)

# Stage 2: Deconvolve
deconv = scmapst.deconvolve(
    stage1_model_path=results['model_path'],
    st_file="data/st.h5ad",
    output_dir="output/stage2/"
)
```

**Command Line Interface:**
```bash
scmapst-train --sc_file data/sc.h5ad --st_file data/st.h5ad --output_dir output/
scmapst-deconvolve --stage1_model output/vae_best_model.pth --st_file data/st.h5ad
```

### 2. Installation

```bash
# Quick install
pip install -e .

# Or use setup script
bash setup.sh
```

### 3. Backward Compatibility

The original `SC_MAP_ST/` code remains unchanged and is used internally. Your existing scripts and notebooks will continue to work.

## Installation Steps

1. **Install package:**
   ```bash
   cd /mnt/d/ST_Graduation_Project
   pip install -e .
   ```

2. **Verify installation:**
   ```bash
   python test_installation.py
   ```

3. **Run example:**
   ```bash
   python example_usage.py
   ```

## Migration Guide

### From os.system() calls to API

**Before:**
```python
import os

cmd = f"python SC_MAP_ST/stage1.py --sc_file {sc_file} --st_file {st_file} --output_dir {output}"
os.system(cmd)
```

**After:**
```python
import scmapst

results = scmapst.train(
    sc_file=sc_file,
    st_file=st_file,
    output_dir=output
)
```

**Benefits:**
- ✅ No string formatting errors
- ✅ Direct access to results
- ✅ Better error messages
- ✅ Type hints and IDE support
- ✅ No subprocess overhead

### Batch Processing Example

```python
import scmapst
from pathlib import Path

# Train reference model once
stage1_results = scmapst.train(
    sc_file="reference/sc.h5ad",
    st_file="reference/st.h5ad",
    output_dir="models/"
)

# Apply to multiple samples
for st_file in Path("data/samples/").glob("*.h5ad"):
    sample_name = st_file.stem
    
    stage2_results = scmapst.deconvolve(
        stage1_model_path=stage1_results['model_path'],
        st_file=str(st_file),
        output_dir=f"results/{sample_name}/"
    )
    
    print(f"{sample_name}: Pearson={stage2_results['best_pearson']:.4f}")
```

## API Reference

### scmapst.train()

Train VAE model for SC-ST integration (Stage 1).

**Parameters:**
- `sc_file` (str): Path to single-cell h5ad file
- `st_file` (str): Path to spatial transcriptomics h5ad file
- `output_dir` (str): Output directory path
- `n_epochs` (int): Number of training epochs (default: 150)
- `resolution` (float): Leiden clustering resolution (default: 4.0)
- `top_n_per_type` (int): Marker genes per cluster (default: 100)
- `use_dual_decoder` (bool): Use DualDecoderVAE (default: True)
- `device` (str): 'cuda', 'cpu', or None for auto-select

**Returns:**
- Dictionary with keys: `model_path`, `n_clusters`, `n_genes`, `best_loss`

### scmapst.deconvolve()

Run GAT-based spatial deconvolution (Stage 2).

**Parameters:**
- `stage1_model_path` (str): Path to trained Stage 1 VAE model
- `st_file` (str): Path to spatial transcriptomics h5ad file
- `output_dir` (str): Output directory path
- `n_epochs` (int): Number of training epochs (default: 200)
- `scale_basis` (str): Gene set for scaling ('all', 'marker', 'hvg', 'none')
- `k_spatial` (int): Spatial neighbors for GAT (default: 20)
- `device` (str): 'cuda', 'cpu', or None for auto-select

**Returns:**
- Dictionary with keys: `best_pearson`, `best_mse`, `best_cosine`, `sample_name`, `deconv_weights_path`

### scmapst.analyze_cellchat()

Analyze cell-cell communication (Stage 3) - **Placeholder, not yet implemented**.

## Files Created

### Core Package Files
- [x] `scmapst/__init__.py` - Main package entry point
- [x] `scmapst/__version__.py` - Version metadata
- [x] `scmapst/cli.py` - Command-line interface
- [x] `scmapst/training/__init__.py` - Training module
- [x] `scmapst/training/stage1.py` - Stage 1 API wrapper
- [x] `scmapst/training/stage2.py` - Stage 2 API wrapper
- [x] `scmapst/training/stage3.py` - Stage 3 placeholder
- [x] `scmapst/models/__init__.py` - Models placeholder
- [x] `scmapst/preprocessing/__init__.py` - Preprocessing placeholder
- [x] `scmapst/utils/__init__.py` - Utils placeholder

### Installation Files
- [x] `setup.py` - Setuptools configuration
- [x] `pyproject.toml` - Modern Python package config
- [x] `MANIFEST.in` - Package data rules

### Documentation Files
- [x] `README.md` - Main documentation
- [x] `INSTALL.md` - Installation guide
- [x] `PACKAGE_SUMMARY.md` - This file

### Testing Files
- [x] `example_usage.py` - Usage example
- [x] `test_installation.py` - Installation test
- [x] `setup.sh` - Quick setup script

## Next Steps

### Immediate (Required)

1. **Install and test the package:**
   ```bash
   bash setup.sh
   ```

2. **Verify with your data:**
   ```python
   import scmapst
   
   results = scmapst.train(
       sc_file="your_sc.h5ad",
       st_file="your_st.h5ad",
       output_dir="test_output/"
   )
   ```

3. **Update notebooks to use API:**
   - Replace `os.system("python stage1.py ...")` with `scmapst.train(...)`
   - Simplify batch processing loops

### Short-term (Recommended)

1. **Add tests:**
   - Create `tests/` directory
   - Add unit tests for API functions
   - Add integration tests with mock data

2. **Improve documentation:**
   - Add docstring examples to all functions
   - Create tutorials for common use cases
   - Document parameter tuning guidelines

3. **Add visualization utilities:**
   - Move plotting functions to `scmapst/utils/`
   - Create high-level visualization API

### Long-term (Optional)

1. **Refactor models:**
   - Move models from `SC_MAP_ST/deconv_model.py` to `scmapst/models/`
   - Separate VAE, GAT, and loss functions

2. **Add Stage 3 (CellChat):**
   - Implement cell-cell communication analysis
   - Integrate with CellChat or similar tools

3. **Publish to PyPI:**
   - Test on fresh environments
   - Create release workflow
   - Publish: `python -m build && twine upload dist/*`

## Troubleshooting

### ImportError: "No module named 'scmapst'"
```bash
pip install -e .
```

### CLI commands not found
```bash
pip install -e .  # Reinstall to register entry points
```

### Import errors from SC_MAP_ST
This is normal - the API wrappers handle imports internally. Just use:
```python
import scmapst
```

### Memory issues in batch processing
- Reduce `batch_size`: 512 → 256
- Lower `resolution`: 4.0 → 2.0
- Use `scale_basis='marker'` instead of `'all'`

## Contact

For issues or questions:
1. Check `README.md` and `INSTALL.md`
2. Run `python test_installation.py`
3. Review error messages carefully

## Summary

✓ Package structure created  
✓ API wrappers implemented  
✓ CLI interface added  
✓ Installation files configured  
✓ Documentation written  
✓ Example scripts provided  

**Status**: Ready for installation and testing!

**Next action**: Run `bash setup.sh` to install and test the package.
