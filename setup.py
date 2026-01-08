"""Setup script for the Spagraph package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version from __version__.py
version_dict = {}
with open(Path(__file__).parent / "spagraph" / "__version__.py") as f:
    exec(f.read(), version_dict)

# Read README if exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Spagraph: VAE + GAT spatial transcriptomics deconvolution and cell communication"

setup(
    name="spagraph",
    version=version_dict["__version__"],
    author=version_dict.get("__author__", "Your Name"),
    author_email=version_dict.get("__email__", "your.email@example.com"),
    description="Spagraph: VAE + GAT spatial transcriptomics deconvolution and cell communication",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spagraph",  # Update with actual repo
    packages=find_packages(exclude=["tests", "notebooks", "trash", "results", "notebook"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "umap-learn>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
