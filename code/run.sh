#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
echo "Downloading imagery from Hugging Face..."

# Download only the Imagery folder from the dataset repository
huggingface-cli download MuriloHSO/Beyond-Spatial-Resolution-Code \
    --repo-type dataset \
    --include "Imagery/*" \
    --local-dir /tmp/hf_download

# Move the downloaded subfolders directly into /data/Imagery/
mv /tmp/hf_download/Imagery/PS/* /data/Imagery/PS/
mv /tmp/hf_download/Imagery/S2/* /data/Imagery/S2/

echo "Running experiments..."
python3 run.py "$@"
