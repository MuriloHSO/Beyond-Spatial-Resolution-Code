#!/usr/bin/env bash
set -ex

# Beyond Spatial Resolution — Code Ocean entry point
echo "Downloading imagery from Hugging Face..."

# 1. Run inline python to download the Imagery folder
python3 -c "
from huggingface_hub import snapshot_download

try:
    local_dir = snapshot_download(
        repo_id="MuriloHSO/Beyond-Spatial-Resolution-Code",
        repo_type="dataset",
    )
    print("Downloaded to:", local_dir)
except Exception as e:
    import traceback
    traceback.print_exc()
"

# 2. Copy the files into the relative data directory one level up (../data)
if [ -d "/tmp/hf_download/Imagery/PS" ]; then
    cp -r /tmp/hf_download/Imagery/PS/. ../data/Imagery/PS/
    cp -r /tmp/hf_download/Imagery/S2/. ../data/Imagery/S2/
elif [ -d "/tmp/hf_download/imagery/ps" ]; then
    cp -r /tmp/hf_download/imagery/ps/. ../data/Imagery/PS/
    cp -r /tmp/hf_download/imagery/s2/. ../data/Imagery/S2/
fi

echo "Running experiments..."
python3 run.py "$@"