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

# 2. Copy the files inside the downloaded subfolders into the relative data directories (no leading slash)
if [ -d "/tmp/hf_download/Imagery/PS" ]; then
    cp -r /tmp/hf_download/Imagery/PS/. data/Imagery/PS/
    cp -r /tmp/hf_download/Imagery/S2/. data/Imagery/S2/
elif [ -d "/tmp/hf_download/imagery/ps" ]; then
    cp -r /tmp/hf_download/imagery/ps/. data/Imagery/PS/
    cp -r /tmp/hf_download/imagery/s2/. data/Imagery/S2/
fi

# 3. Debug line: Print the contents to verify they are in the relative workspace path
echo "Checking relative target data directories:"
ls -la data/Imagery/PS/
ls -la data/Imagery/S2/

# Force stdout to flush immediately so we can see the debug logs in order
export PYTHONUNBUFFERED=1

echo "Running experiments..."
python3 run.py "$@"