#!/bin/bash
# Quick setup script for scmapst package

set -e  # Exit on error

echo "============================================================"
echo "SC-MAP-ST Package Setup"
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
echo "Installing scmapst package (editable mode)..."
echo "============================================================"
pip install -e .

if [ $? -eq 0 ]; then
    echo "✓ Package installed successfully"
else
    echo "❌ Installation failed"
    exit 1
fi

# Run installation tests
echo ""
echo "============================================================"
echo "Running installation tests..."
echo "============================================================"
python test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Setup completed successfully!"
    echo "============================================================"
    echo ""
    echo "You can now use scmapst:"
    echo "  • Python API:  import scmapst"
    echo "  • CLI:         scmapst-train --help"
    echo "  • Example:     python example_usage.py"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "⚠ Setup completed with warnings"
    echo "============================================================"
    echo "Package is installed but some tests failed."
    echo "Check the output above for details."
    echo ""
fi
