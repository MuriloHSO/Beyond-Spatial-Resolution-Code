"""
data.py
-------
Dataset loading utilities.
"""

import pandas as pd


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

    df_s2train = pd.read_csv(datasets_path / "S2_Training.csv").dropna()
    df_s2val = pd.read_csv(datasets_path / "S2_Validation.csv").dropna()
    df_psstrain = pd.read_csv(datasets_path / "PS_Training.csv").dropna()
    df_pssval = pd.read_csv(datasets_path / "PS_Validation.csv").dropna()

    return {
        "df_s2train": df_s2train,
        "df_s2val": df_s2val,
        "df_psstrain": df_psstrain,
        "df_pssval": df_pssval,
    }
