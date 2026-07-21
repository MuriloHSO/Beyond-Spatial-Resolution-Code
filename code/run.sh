#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
echo "Downloading imagery from Hugging Face..."

# Run a quick inline python command to download the exact Imagery folder structure
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MuriloHSO/Beyond-Spatial-Resolution-Code',
    repo_type='dataset',
    allow_patterns='Imagery/*',
    local_dir='/tmp/hf_download'
)
"

# Move the downloaded subfolder contents into their matching data folders
mv /tmp/hf_download/Imagery/PS/* /data/Imagery/PS/
mv /tmp/hf_download/Imagery/S2/* /data/Imagery/S2/

echo "Running experiments..."
python3 run.py "$@"