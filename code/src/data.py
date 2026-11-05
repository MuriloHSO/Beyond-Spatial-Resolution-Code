"""
data.py
-------
Dataset loading utilities.
"""

import pandas as pd

# PS point tables: not distributed (Planet's licence; see README), so they may be absent.
PS_TABLES = ("PS_TrainTest.csv", "PS_Validation.csv")


def ps_tables_present(datasets_path) -> bool:
    """True if both PS point tables are in *datasets_path*."""
    return all((datasets_path / name).exists() for name in PS_TABLES)


def _read(path) -> pd.DataFrame:
    """One point table; rows with a missing value are dropped (Polygon_ID is empty for "others" by design)."""
    df = pd.read_csv(path)
    return df.dropna(subset=[c for c in df.columns if c != "Polygon_ID"])


def load_datasets(paths: dict) -> dict:
    """
    Load the CSV datasets from disk; the PS tables only if both are present.

    Parameters
    ----------
    paths : dict
        Output of :func:`src.paths.setup_paths`.

    Returns
    -------
    dict with keys:
        df_s2train, df_s2val, and df_psstrain, df_pssval when the PS tables are present
    """
    datasets_path = paths["datasets_path"]

    datasets = {
        "df_s2train": _read(datasets_path / "S2_TrainTest.csv"),
        "df_s2val": _read(datasets_path / "S2_Validation.csv"),
    }
    if ps_tables_present(datasets_path):
        datasets["df_psstrain"] = _read(datasets_path / "PS_TrainTest.csv")
        datasets["df_pssval"] = _read(datasets_path / "PS_Validation.csv")
    return datasets


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
        ("S2", "Train-test"): "df_s2train",
        ("S2", "Validation"): "df_s2val",
        ("PS", "Train-test"): "df_psstrain",
        ("PS", "Validation"): "df_pssval",
    }
    rows = []
    for (sensor, subset), key in subsets.items():
        if key not in datasets:
            continue
        for label, n in datasets[key]["Class"].value_counts().sort_index().items():
            rows.append([sensor, subset, label, int(n)])
    return pd.DataFrame(rows, columns=["Sensor", "Subset", "Class", "Points"])
