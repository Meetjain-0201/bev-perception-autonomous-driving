#!/bin/bash

echo "=================================================="
echo "Downloading nuScenes Mini Dataset"
echo "=================================================="
echo "Size: ~4GB"
echo "Scenes: 10 (5 train, 5 val)"
echo "Cameras: 6 per frame"
echo "=================================================="

# Create directory
mkdir -p data/nuscenes
cd data/nuscenes

# Download mini dataset
echo "Downloading v1.0-mini.tgz..."
wget https://www.nuscenes.org/data/v1.0-mini.tgz

# Extract
echo "Extracting dataset..."
tar -xzf v1.0-mini.tgz

# Remove archive to save space
echo "Cleaning up..."
rm v1.0-mini.tgz

echo "=================================================="
echo "✅ Dataset downloaded successfully!"
echo "Location: $(pwd)"
echo "=================================================="
echo ""
echo "Dataset structure:"
ls -lh

cd ../..
