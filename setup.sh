#!/bin/bash
# Quick setup script for spagraph package

set -e  # Exit on error

echo "============================================================"
echo "Spagraph Package Setup"
echo "============================================================"

# Check Python version
echo ""
echo "Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python not found. Please install Python >=3.8"
    exit 1
fi

# Check if pip is available
echo ""
echo "Checking pip..."
pip --version
if [ $? -ne 0 ]; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

# Install package in editable mode
echo ""
echo "============================================================"
echo "Installing spagraph package (editable mode)..."
echo "============================================================"
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ Package installed successfully"
else
    echo "❌ Installation failed"
    exit 1
fi

echo ""
echo "============================================================"
echo "Setup completed!"
echo "============================================================"
echo ""
echo "You can now use spagraph:"
echo "  • Python API:  import spagraph"
echo "  • Example:     python example_usage.py"
echo ""
