"""
experiments.py
--------------
Experiment definitions and the dispatcher that runs them.
"""

import pandas as pd

from .training import train_and_evaluate_model


# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------

S2_BANDS_4 = [
    "B2_0", "B3_0", "B4_0", "B8_0",
    "B2_1", "B3_1", "B4_1", "B8_1",
    "B2_2", "B3_2", "B4_2", "B8_2",
]

S2_BANDS_ALL = [
    "B2_0",  "B3_0",  "B4_0",  "B5_0",  "B6_0",  "B7_0",  "B8_0",  "B8A_0", "B11_0", "B12_0",
    "B2_1",  "B3_1",  "B4_1",  "B5_1",  "B6_1",  "B7_1",  "B8_1",  "B8A_1", "B11_1", "B12_1",
    "B2_2",  "B3_2",  "B4_2",  "B5_2",  "B6_2",  "B7_2",  "B8_2",  "B8A_2", "B11_2", "B12_2",
]

PS_BANDS_4 = [
    "B2_0", "B4_0", "B6_0", "B8_0",
    "B2_1", "B4_1", "B6_1", "B8_1",
    "B2_2", "B4_2", "B6_2", "B8_2",
]

PS_BANDS_ALL = [
    "B1_0", "B2_0", "B3_0", "B4_0", "B5_0", "B6_0", "B7_0", "B8_0",
    "B1_1", "B2_1", "B3_1", "B4_1", "B5_1", "B6_1", "B7_1", "B8_1",
    "B1_2", "B2_2", "B3_2", "B4_2", "B5_2", "B6_2", "B7_2", "B8_2",
]


# ---------------------------------------------------------------------------
# Individual experiment runners
# ---------------------------------------------------------------------------

def run_experiment_S2_4b(models, datasets, paths, apply_model_on_image=True, save_excel=True):
    X,  y  = datasets["df_s2train"][S2_BANDS_4], datasets["df_s2train"]["Class"]
    X2, y2 = datasets["df_s2val"][S2_BANDS_4],   datasets["df_s2val"]["Class"]

    imagery = paths["S2_images"][0] if paths["S2_images"] else None
    params, results = train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        apply_model_on_image=apply_model_on_image,
        condition_label="S2_1",
    )

    if save_excel:
        results.to_excel(paths["results_path"] / "S2_4b.xlsx", index=False)

    return params, results


def run_experiment_S2_Allb(models, datasets, paths, apply_model_on_image=True, save_excel=True):
    X,  y  = datasets["df_s2train"][S2_BANDS_ALL], datasets["df_s2train"]["Class"]
    X2, y2 = datasets["df_s2val"][S2_BANDS_ALL],   datasets["df_s2val"]["Class"]

    imagery = paths["S2_images"][0] if paths["S2_images"] else None
    params, results = train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        apply_model_on_image=apply_model_on_image,
        condition_label="S2_2",
    )

    if save_excel:
        results.to_excel(paths["results_path"] / "S2_Allb.xlsx", index=False)

    return params, results


def run_experiment_PS_4b(models, datasets, paths, apply_model_on_image=False, save_excel=True):
    X,  y  = datasets["df_psstrain"][PS_BANDS_4], datasets["df_psstrain"]["Class"]
    X2, y2 = datasets["df_pssval"][PS_BANDS_4],   datasets["df_pssval"]["Class"]

    imagery = paths["PS_images"][0] if paths["PS_images"] else None
    params, results = train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        apply_model_on_image=apply_model_on_image,
        condition_label="PS_1",
    )

    if save_excel:
        results.to_excel(paths["results_path"] / "PS_4b.xlsx", index=False)

    return params, results


def run_experiment_PS_Allb(models, datasets, paths, apply_model_on_image=True, save_excel=True):
    X,  y  = datasets["df_psstrain"][PS_BANDS_ALL], datasets["df_psstrain"]["Class"]
    X2, y2 = datasets["df_pssval"][PS_BANDS_ALL],   datasets["df_pssval"]["Class"]

    imagery = paths["PS_images"][0] if paths["PS_images"] else None
    params, results = train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        apply_model_on_image=apply_model_on_image,
        condition_label="PS_2",
    )

    if save_excel:
        results.to_excel(paths["results_path"] / "PS_Allb.xlsx", index=False)

    return params, results


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENT_QUEUE = {
    "S2_4b":   run_experiment_S2_4b,
    "S2_Allb": run_experiment_S2_Allb,
    "PS_4b":   run_experiment_PS_4b,
    "PS_Allb": run_experiment_PS_Allb,
}


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def run_all_experiments(selected_experiments: list, models: dict, datasets: dict, paths: dict) -> pd.DataFrame:
    """
    Run a list of experiments and return the combined results DataFrame.

    Parameters
    ----------
    selected_experiments : list of dict
        Each dict must have:
            - 'name'  : str — key in EXPERIMENT_QUEUE
            - 'apply_model_on_image' : bool (optional, default True)
    models : dict
        Output of :func:`src.config.build_models`.
    datasets : dict
        Output of :func:`src.data.load_datasets`.
    paths : dict
        Output of :func:`src.paths.setup_paths`.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all experiments, also saved to
        ``results_path/model_results.xlsx``.
    """
    selected_results = []
    n_exps = len(selected_experiments)

    for exp_i, item in enumerate(selected_experiments, 1):
        name = item["name"]
        apply_flag = item.get("apply_model_on_image", True)

        if name not in EXPERIMENT_QUEUE:
            raise ValueError(
                f"Unknown experiment '{name}'. Valid options: {list(EXPERIMENT_QUEUE.keys())}"
            )

        img_note = "  [+image]" if apply_flag else ""
        print(f"\n  + [{exp_i}/{n_exps}] {name}{img_note}")

        _params, results_i = EXPERIMENT_QUEUE[name](
            models=models,
            datasets=datasets,
            paths=paths,
            apply_model_on_image=apply_flag,
            save_excel=False,
        )
        selected_results.append(results_i)

    combined = (
        pd.concat(selected_results, ignore_index=True)
        if selected_results
        else pd.DataFrame()
    )
    combined.to_excel(paths["results_path"] / "model_results.xlsx", index=False)
    return combined
