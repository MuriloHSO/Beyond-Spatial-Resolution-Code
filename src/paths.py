"""
paths.py
--------
All filesystem path definitions and directory creation.
"""

import glob
from pathlib import Path


def setup_paths(base: Path = None) -> dict:
    """
    Resolve every path used by the project and create output directories.

    Parameters
    ----------
    base : Path, optional
        Root of the project.  Defaults to the parent of this file (i.e. the
        repository root when the package lives in ``<repo>/src/``).

    Returns
    -------
    dict with keys:
        base_path, datasets_path, images_path, results_path,
        S2_imagery, PS_imagery, S2_images, PS_images
    """
    if base is None:
        # src/ lives one level below the repo root
        base = Path(__file__).resolve().parent.parent

    datasets_path = base / "data"
    images_path = base / "data" / "Imagery"
    results_path = base / "results"

    # Create output sub-directories
    (results_path / "PNG").mkdir(parents=True, exist_ok=True)
    (results_path / "TIFF").mkdir(parents=True, exist_ok=True)

    S2_imagery = images_path / "S2"
    PS_imagery = images_path / "PS"
    
    # Try subfolders first, fallback to root data/ folder matching prefix
    S2_images = glob.glob(str(S2_imagery / "*.tif"))
    if not S2_images:
        S2_images = glob.glob(str(images_path / "S2*.tif"))
        
    PS_images = glob.glob(str(PS_imagery / "*.tif"))
    if not PS_images:
        PS_images = glob.glob(str(images_path / "PS*.tif"))

    return {
        "base_path": base,
        "datasets_path": datasets_path,
        "images_path": images_path,
        "results_path": results_path,
        "S2_imagery": S2_imagery,
        "PS_imagery": PS_imagery,
        "S2_images": S2_images,
        "PS_images": PS_images,
    }
