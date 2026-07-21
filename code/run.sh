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

# 2. Check if folders are uppercase or lowercase in /tmp and copy contents
if [ -d "/tmp/hf_download/Imagery/PS" ]; then
    cp -r /tmp/hf_download/Imagery/PS/. /data/Imagery/PS/
    cp -r /tmp/hf_download/Imagery/S2/. /data/Imagery/S2/
elif [ -d "/tmp/hf_download/imagery/ps" ]; then
    cp -r /tmp/hf_download/imagery/ps/. /data/Imagery/PS/
    cp -r /tmp/hf_download/imagery/s2/. /data/Imagery/S2/
fi

# 3. Debug line: Print the contents of the target folder to ensure the files are there
echo "Checking target data directories:"
ls -la /data/Imagery/PS/
ls -la /data/Imagery/S2/

echo "Running experiments..."
python3 run.py "$@"