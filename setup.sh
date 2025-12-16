#!/bin/bash

echo "=========================================="
echo "YouTube Comments Sentiment - Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv youtube-sentiment

# Activate virtual environment
echo "Activating virtual environment..."
source youtube-sentiment/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p artifacts
mkdir -p data

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="