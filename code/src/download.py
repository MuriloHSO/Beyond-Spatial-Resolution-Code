"""
download.py

Download the imagery dataset from Hugging Face if it is not already
available locally.
"""

from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

from src.paths import get_image_lists


def ensure_imagery(paths: dict, force_download: bool = False) -> None:
    """
    Ensure Sentinel-2 and PlanetScope imagery are available.

    Parameters
    ----------
    paths : dict
        Dictionary returned by setup_paths().

    force_download : bool, default=False
        If True, always re-download the imagery.
    """

    images = get_image_lists(paths)

    if (
        not force_download
        and len(images["S2"]) > 0
        and len(images["PS"]) > 0
    ):
        print("[OK] Imagery already available.")
        print(f"  Sentinel-2 : {len(images['S2'])} images")
        print(f"  PlanetScope: {len(images['PS'])} images")
        return

    print("\nDownloading imagery from Hugging Face...")

    snapshot_path = Path(
        snapshot_download(
            repo_id="MuriloHSO/Beyond-Spatial-Resolution-Code",
            repo_type="dataset",
            allow_patterns="Imagery/**",
        )
    )

    imagery_source = snapshot_path / "Imagery"

    if not imagery_source.exists():
        raise RuntimeError(
            f"Imagery directory not found:\n{imagery_source}"
        )

    print(f"Downloaded snapshot:\n{snapshot_path}")

    #
    # Destination directories
    #

    s2_destination = paths["S2_imagery"]
    ps_destination = paths["PS_imagery"]

    s2_destination.mkdir(parents=True, exist_ok=True)
    ps_destination.mkdir(parents=True, exist_ok=True)

    #
    # Copy imagery
    #

    print("Copying Sentinel-2 imagery...")

    shutil.copytree(
        imagery_source / "S2",
        s2_destination,
        dirs_exist_ok=True,
    )

    print("Copying PlanetScope imagery...")

    shutil.copytree(
        imagery_source / "PS",
        ps_destination,
        dirs_exist_ok=True,
    )

    #
    # Verify
    #

    images = get_image_lists(paths)

    print("\nImagery successfully installed.")

    print(f"  Sentinel-2 : {len(images['S2'])} images")
    print(f"  PlanetScope: {len(images['PS'])} images")

    if len(images["S2"]) == 0:
        raise RuntimeError(
            "Sentinel-2 imagery was not found after download."
        )

    if len(images["PS"]) == 0:
        raise RuntimeError(
            "PlanetScope imagery was not found after download."
        )