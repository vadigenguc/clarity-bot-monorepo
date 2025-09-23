#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting custom build script..."

# 1. Install core dependencies
echo "Installing core Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# 2. Install sentence-transformers separately and download the model
# This helps avoid OOM errors during the main pip install
echo "Installing sentence-transformers..."
pip install --no-cache-dir sentence-transformers

# Define the model name and a custom cache directory
MODEL_NAME="all-MiniLM-L6-v2"
MODEL_CACHE_DIR="./.cache/sentence_transformers"

echo "Custom build script finished."
# Model download will now happen at runtime in main.py if not present.
