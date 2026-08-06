"""
paths.py

Utilities for defining and managing the directory structure used by the
Beyond Spatial Resolution project.
"""

from pathlib import Path


def setup_paths(base: Path | str | None = None) -> dict:
    """
    Create and return all project paths.

    Parameters
    ----------
    base : Path | str | None
        Project root directory.
        If None, uses the parent directory of this file.

    Returns
    -------
    dict
        Dictionary containing all project directories.
    """

    if base is None:
        base = Path(__file__).resolve().parent.parent
    else:
        base = Path(base)

    # ------------------------------------------------------------------
    # Input data
    # ------------------------------------------------------------------

    data_path = base / "data"

    datasets_path = data_path

    images_path = data_path / "Imagery"

    S2_imagery = images_path / "S2"

    PS_imagery = images_path / "PS"

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    results_path = base / "results"

    figures_path = results_path / "figures"

    maps_path = results_path / "maps"

    maps_png_path = maps_path / "PNG"

    maps_tif_path = maps_path / "TIFF"

    models_path = results_path / "Models"

    logs_path = results_path / "Logs"

    metadata_path = results_path / "Metadata"

    # ------------------------------------------------------------------
    # Create directories
    # ------------------------------------------------------------------

    directories = [
        data_path,
        datasets_path,
        images_path,
        S2_imagery,
        PS_imagery,
        results_path,
        figures_path,
        maps_path,
        maps_png_path,
        maps_tif_path,
        models_path,
        logs_path,
        metadata_path,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,

        "data_path": data_path,
        "datasets_path": datasets_path,

        "images_path": images_path,
        "S2_imagery": S2_imagery,
        "PS_imagery": PS_imagery,

        "results_path": results_path,
        "figures_path": figures_path,

        "maps_path": maps_path,
        "maps_png_path": maps_png_path,
        "maps_tif_path": maps_tif_path,

        "models_path": models_path,
        "logs_path": logs_path,
        "metadata_path": metadata_path,
    }


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def get_image_lists(paths: dict) -> dict:
    """
    Return the imagery currently available on disk.

    Returns
    -------
    dict

        {
            "S2": [...],
            "PS": [...]
        }
    """

    return {
        "S2": sorted(paths["S2_imagery"].glob("*.tif")),
        "PS": sorted(paths["PS_imagery"].glob("*.tif")),
    }


def count_images(paths: dict) -> dict:
    """
    Count available images.

    Returns
    -------
    dict
        {"S2": n, "PS": n}
    """

    images = get_image_lists(paths)

    return {
        key: len(value)
        for key, value in images.items()
    }
