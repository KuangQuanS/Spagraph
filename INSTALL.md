# SC-MAP-ST Package Installation Guide

## Quick Installation

### 1. Install the package in editable mode

```bash
# Navigate to the project root
cd /mnt/d/ST_Graduation_Project

# Install in editable mode (recommended for development)
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### 2. Verify installation

```bash
# Test Python import
python -c "import scmapst; print(scmapst.__version__)"

# Test CLI commands
scmapst-train --help
scmapst-deconvolve --help
```

## Usage Examples

### Python API

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
deconv_results = scmapst.deconvolve(
    stage1_model_path=results['model_path'],
    st_file="data/st.h5ad",
    output_dir="output/stage2/"
)
```

### Command Line

```bash
# Stage 1: Train VAE
scmapst-train \
    --sc_file data/sc.h5ad \
    --st_file data/st.h5ad \
    --output_dir output/stage1/ \
    --n_epochs 150 \
    --resolution 4.0

# Stage 2: Deconvolve
scmapst-deconvolve \
    --stage1_model output/stage1/vae_best_model.pth \
    --st_file data/st.h5ad \
    --output_dir output/stage2/ \
    --scale_basis marker
```

## Testing with Existing Notebooks

Your existing notebooks can now use the package:

```python
# In Jupyter notebook
import scmapst

# Replace this:
# os.system("python ../SC_MAP_ST/stage1.py --sc_file ... --st_file ...")

# With this:
results = scmapst.train(
    sc_file=sc_path,
    st_file=st_path,
    output_dir=output_dir,
    n_epochs=150,
    resolution=4.0
)
```

## Migrating from run_simulate.ipynb

### Old approach (using os.system):
```python
import os

for dataset in datasets:
    cmd = f"python ../SC_MAP_ST/stage1.py --sc_file {sc_file} --st_file {st_file} ..."
    os.system(cmd)
    
    cmd = f"python ../SC_MAP_ST/stage2.py --stage1_model {model_path} ..."
    os.system(cmd)
```

### New approach (using scmapst API):
```python
import scmapst

for dataset in datasets:
    # Stage 1
    stage1_results = scmapst.train(
        sc_file=sc_file,
        st_file=st_file,
        output_dir=f"results/{dataset}/stage1/"
    )
    
    # Stage 2
    stage2_results = scmapst.deconvolve(
        stage1_model_path=stage1_results['model_path'],
        st_file=st_file,
        output_dir=f"results/{dataset}/stage2/"
    )
```

Benefits:
- ✅ Cleaner code, no string formatting
- ✅ Direct access to results (no parsing output files)
- ✅ Better error handling
- ✅ Type hints and IDE autocompletion
- ✅ No subprocess overhead

## Troubleshooting

### Import Error: "No module named 'scmapst'"

Make sure you're in the correct environment and installed the package:
```bash
pip install -e .
```

### Import Error: "No module named 'stage1'"

This is expected - the old `SC_MAP_ST/` modules are imported internally. 
Just use the new API: `import scmapst`

### CUDA/GPU Issues

If you encounter CUDA errors:
```python
# Force CPU usage
results = scmapst.train(..., device='cpu')
```

## Package Structure

```
scmapst/
├── __init__.py           # Main package entry (exports train, deconvolve, analyze_cellchat)
├── __version__.py        # Version information
├── cli.py               # Command-line interface
├── training/
│   ├── __init__.py
│   ├── stage1.py        # API wrapper for Stage 1
│   ├── stage2.py        # API wrapper for Stage 2
│   └── stage3.py        # Placeholder for CellChat
├── models/              # (Reserved for model definitions)
├── preprocessing/       # (Reserved for data preprocessing)
└── utils/              # (Reserved for utility functions)
```

## Next Steps

1. Test with your existing data:
   ```bash
   python example_usage.py
   ```

2. Update your notebooks to use the new API

3. Run batch processing with simplified code:
   ```python
   import scmapst
   from pathlib import Path
   
   for st_file in Path("data/").glob("*.h5ad"):
       results = scmapst.train(...)
       deconv = scmapst.deconvolve(...)
   ```

4. Consider publishing to PyPI for wider distribution:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```
