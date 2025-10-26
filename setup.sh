#!/bin/bash

# Setup script for Astronomical Object Classifier
# This script creates a Python 3.10 virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Astronomical Object Classifier Setup"
echo "=========================================="

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: python3.10 is not installed"
    echo "Please install Python 3.10 first:"
    echo "  - macOS: brew install python@3.10"
    echo "  - Ubuntu: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

echo "✓ Python 3.10 found: $(python3.10 --version)"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo ""
    echo "Virtual environment 'venv' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
        exit 0
    fi
fi

# Create virtual environment
echo ""
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "For GPU support (when available), install PyTorch with CUDA:"
echo "  https://pytorch.org/get-started/locally/"
echo ""

