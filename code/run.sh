#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
echo "Downloading imagery from Hugging Face..."

# 1. Download only the Imagery folder from the dataset repository using the 'hf' tool
hf download MuriloHSO/Beyond-Spatial-Resolution-Code \
    --include "Imagery/*" \
    --local-dir /tmp/hf_download

# 2. Move the downloaded subfolder contents into their matching data folders
mv /tmp/hf_download/Imagery/PS/* /data/Imagery/PS/
mv /tmp/hf_download/Imagery/S2/* /data/Imagery/S2/

echo "Running experiments..."
python3 run.py "$@"