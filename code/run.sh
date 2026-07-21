#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
echo "Downloading imagery from Hugging Face..."

# 1. Run inline python to download the Imagery folder
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MuriloHSO/Beyond-Spatial-Resolution-Code',
    repo_type='dataset',
    allow_patterns='Imagery/*',
    local_dir='/tmp/hf_download'
)
"

# 2. Safely clear out the placeholder folders and move the downloaded data into place
rm -rf /data/Imagery/PS /data/Imagery/S2
mv /tmp/hf_download/Imagery/PS /data/Imagery/PS
mv /tmp/hf_download/Imagery/S2 /data/Imagery/S2

echo "Running experiments..."
python3 run.py "$@"