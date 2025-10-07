#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."
echo "Python version:"
python --version

# Update pip and setuptools first
echo "Updating pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install requirements with no build isolation
echo "Installing requirements..."
pip install --no-build-isolation --no-cache-dir -r requirements.txt

echo "Build completed successfully!"
