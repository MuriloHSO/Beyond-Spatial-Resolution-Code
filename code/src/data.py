"""
data.py
-------
Dataset loading utilities.
"""

import pandas as pd


def _read(path) -> pd.DataFrame:
    """One point table; rows with a missing value are dropped (Polygon_ID is empty for "others" by design)."""
    df = pd.read_csv(path)
    return df.dropna(subset=[c for c in df.columns if c != "Polygon_ID"])


def load_datasets(paths: dict) -> dict:
    """
    Load the four CSV datasets from disk.

    Parameters
    ----------
    paths : dict
        Output of :func:`src.paths.setup_paths`.

    Returns
    -------
    dict with keys:
        df_s2train, df_s2val, df_psstrain, df_pssval
    """
    datasets_path = paths["datasets_path"]

    df_s2train = _read(datasets_path / "S2_Training.csv")
    df_s2val = _read(datasets_path / "S2_Validation.csv")
    df_psstrain = _read(datasets_path / "PS_Training.csv")
    df_pssval = _read(datasets_path / "PS_Validation.csv")

    return {
        "df_s2train": df_s2train,
        "df_s2val": df_s2val,
        "df_psstrain": df_psstrain,
        "df_pssval": df_pssval,
    }


def sample_counts(datasets: dict) -> pd.DataFrame:
    """
    Number of sample points per sensor, subset and class (after removing rows with missing values).

    Parameters
    ----------
    datasets : dict
        Output of :func:`load_datasets`.

    Returns
    -------
    pd.DataFrame with columns Sensor, Subset, Class, Points
    """
    subsets = {
        ("S2", "Training"): "df_s2train",
        ("S2", "Validation"): "df_s2val",
        ("PS", "Training"): "df_psstrain",
        ("PS", "Validation"): "df_pssval",
    }
    rows = []
    for (sensor, subset), key in subsets.items():
        for label, n in datasets[key]["Class"].value_counts().sort_index().items():
            rows.append([sensor, subset, label, int(n)])
    return pd.DataFrame(rows, columns=["Sensor", "Subset", "Class", "Points"])
