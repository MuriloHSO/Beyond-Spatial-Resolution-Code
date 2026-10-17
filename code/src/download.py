"""
download.py

Download the imagery dataset from Hugging Face if it is not already
available locally.

Run from the `code` folder to download the imagery without running the
experiments:

    python -m src.download
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from src.paths import get_image_lists, setup_paths

REPO_ID = "MuriloHSO/Beyond-Spatial-Resolution-Code"
REVISION = "8e9663b9359d7bd0ff4cdf58faebedb0e5acf923"  # dataset commit "Imagery PS and S2"

# GeoTIFFs of the dataset at REVISION, stored under Imagery/<sensor>/.
IMAGERY_FILES = {
    "S2": [
        "S2_holambra3t_proj_2022-12-15_0.tif",
        "S2_holambra3t_proj_2023-06-18_0.tif",
        "S2_sumare3t_proj_2022-12-15_0.tif",
        "S2_sumare3t_proj_2023-06-18_0.tif",
    ],
    "PS": [
        "PS_Holambra3t_proj_2022-12-15_0.tif",
        "PS_holambra3t_proj_2023-06-18_0.tif",
        "PS_sumare3t_proj_2022-12-15_0.tif",
        "PS_sumare3t_proj_2023-06-18_0.tif",
    ],
}


def _missing_files(paths: dict) -> list:
    return [
        paths[f"{sensor}_imagery"] / name
        for sensor, names in IMAGERY_FILES.items()
        for name in names
        if not (paths[f"{sensor}_imagery"] / name).exists()
    ]


def ensure_imagery(paths: dict, force_download: bool = False) -> None:
    """
    Ensure Sentinel-2 and PlanetScope imagery are available.

    The files are downloaded directly into Imagery/S2 and Imagery/PS (no
    copy in the Hugging Face cache); an interrupted download resumes.

    Parameters
    ----------
    paths : dict
        Dictionary returned by setup_paths().

    force_download : bool, default=False
        If True, always re-download the imagery.
    """

    if not force_download and not _missing_files(paths):
        images = get_image_lists(paths)
        print("[OK] Imagery already available.")
        print(f"  Sentinel-2 : {len(images['S2'])} images")
        print(f"  PlanetScope: {len(images['PS'])} images")
        return

    print("\nDownloading imagery from Hugging Face...")

    # The dataset stores the images under Imagery/<sensor>/, so downloading
    # into the parent of images_path places them in the sensor folders.
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns="Imagery/*/*.tif",
        local_dir=paths["images_path"].parent,
        force_download=force_download,
    )

    missing = _missing_files(paths)
    if missing:
        raise RuntimeError(
            "Imagery not found after download:\n"
            + "\n".join(str(path) for path in missing)
        )

    # huggingface_hub keeps download metadata in <local_dir>/.cache/huggingface
    # to resume interrupted downloads; it is not needed once every file is present.
    hf_cache = paths["images_path"].parent / ".cache" / "huggingface"
    shutil.rmtree(hf_cache, ignore_errors=True)
    if hf_cache.parent.exists() and not any(hf_cache.parent.iterdir()):
        hf_cache.parent.rmdir()

    images = get_image_lists(paths)

    print("\nImagery successfully installed.")
    print(f"  Sentinel-2 : {len(images['S2'])} images")
    print(f"  PlanetScope: {len(images['PS'])} images")


if __name__ == "__main__":
    # Repository root, as run.py passes it locally.
    ensure_imagery(setup_paths(base=Path(__file__).resolve().parents[2]))
