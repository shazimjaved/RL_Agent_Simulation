#!/bin/bash
# Build script for Render deployment

echo "Starting build process..."

# Update pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

echo "Build completed successfully!"
