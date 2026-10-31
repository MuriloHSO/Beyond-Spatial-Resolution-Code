"""
download.py

Download the Sentinel-2 composites used for full-image classification from
Hugging Face into data/composites/Sentinel-2. The folder is filled only when it
holds no composite (or an earlier download was interrupted), so composites built
locally by src/preprocessing.py are never replaced. When the folder is read-only
(the /data folder of Code Ocean during a run), the composites are downloaded to
<scratch>/composites/ instead and used from there.

The PlanetScope composites are not distributed: Planet's Education and Research
terms do not allow sharing them. Build them from your own Planet orders as
described in data/composites/PlanetScope/README.md.

Run from the `code` folder to download the composites without running the
experiments:

    python -m src.download
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from src.paths import get_image_lists, setup_paths

REPO_ID = "marlonfs/agricultural-plastic-PS-S2"
REVISION = "15c18f4590376b75a8f9508e19d749081213e2d9"  # dataset commit "Data of the paper: S2 composites, ..."

# Sensors whose composites are distributed on Hugging Face (PS: see the module docstring).
SENSOR_FOLDERS = {"S2": "Sentinel-2"}
ROIS = ["Holambra", "Sumare"]
REFERENCE_DATES = ["2021-12-15", "2022-06-18", "2022-12-15", "2023-06-18"]

# GeoTIFFs of the dataset at REVISION, stored under composites/<sensor folder>/.
COMPOSITE_FILES = {
    sensor: [f"{sensor}_{roi}_{date}.tif" for roi in ROIS for date in REFERENCE_DATES]
    for sensor in SENSOR_FOLDERS
}


def _hf_metadata(root: Path) -> Path:
    # huggingface_hub keeps download metadata in <local_dir>/.cache/huggingface
    # to resume interrupted downloads.
    return root / ".cache" / "huggingface"


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _to_download(paths: dict, sensor: str, force_download: bool) -> bool:
    """True when the sensor folder has no composite or holds an interrupted download."""
    folder = paths[f"{sensor}_composites"]
    interrupted = (_hf_metadata(folder.parents[1]) / "download" / "composites" / SENSOR_FOLDERS[sensor]).exists()
    return force_download or interrupted or not any(folder.glob(f"{sensor}_*.tif"))


def ensure_composites(paths: dict, force_download: bool = False) -> None:
    """
    Ensure the Sentinel-2 composites are available and report the PlanetScope ones.

    An empty Sentinel-2 folder is filled from the Hugging Face dataset (files
    downloaded directly, no copy in the Hugging Face cache); an interrupted
    download resumes. PlanetScope composites are only reported: without them,
    PS full-image classification is skipped.

    Parameters
    ----------
    paths : dict
        Dictionary returned by setup_paths().

    force_download : bool, default=False
        If True, always re-download the composites.
    """

    if not any(paths["PS_composites"].glob("PS_*.tif")):
        print("[!] No PlanetScope composites in data/composites/PlanetScope: they are not distributed "
              "(Planet's Education and Research terms). Build them from your own Planet orders "
              "(data/composites/PlanetScope/README.md); PS full-image classification is skipped.")

    sensors = [sensor for sensor in SENSOR_FOLDERS if _to_download(paths, sensor, force_download)]

    if not sensors:
        images = get_image_lists(paths)
        print("[OK] Composites already available.")
        print(f"  Sentinel-2 : {len(images['S2'])} composites")
        print(f"  PlanetScope: {len(images['PS'])} composites")
        return

    # The dataset stores the composites under composites/<sensor folder>/, so
    # downloading into <root> places them in <root>/composites/<sensor folder>/.
    # <root> is data/, or the scratch folder when data/ is read-only.
    root = paths["data_path"]
    if not all(_writable(paths[f"{sensor}_composites"]) for sensor in sensors):
        root = paths["scratch_path"]
        for sensor in sensors:
            paths[f"{sensor}_composites"] = root / "composites" / SENSOR_FOLDERS[sensor]

    print(f"\nDownloading composites from Hugging Face ({', '.join(SENSOR_FOLDERS[s] for s in sensors)}) "
          f"to {root / 'composites'}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"composites/{SENSOR_FOLDERS[s]}/*.tif" for s in sensors],
        local_dir=root,
        force_download=force_download,
    )

    missing = [
        paths[f"{sensor}_composites"] / name
        for sensor in sensors
        for name in COMPOSITE_FILES[sensor]
        if not (paths[f"{sensor}_composites"] / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Composites not found after download:\n"
            + "\n".join(str(path) for path in missing)
        )

    # The download metadata is not needed once every file is present.
    hf_cache = _hf_metadata(root)
    shutil.rmtree(hf_cache, ignore_errors=True)
    if hf_cache.parent.exists() and not any(hf_cache.parent.iterdir()):
        hf_cache.parent.rmdir()

    images = get_image_lists(paths)

    print("\nComposites successfully installed.")
    print(f"  Sentinel-2 : {len(images['S2'])} composites")
    print(f"  PlanetScope: {len(images['PS'])} composites")


if __name__ == "__main__":
    # Repository root, as run.py passes it locally.
    ensure_composites(setup_paths(base=Path(__file__).resolve().parents[2]))
