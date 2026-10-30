"""
experiments.py
--------------
Experiment definitions and the dispatcher that runs them.
"""

import numpy as np
import pandas as pd
from .config import DEFAULT_RANDOM_STATE
from .training import INTERNAL_COLUMNS, train_and_evaluate_model
from .paths import get_image_lists
from .tuning import polygon_groups
from .uncertainty import confusion_counts, interval, metrics_from_confusion, validation_resamples

# ROI and reference date of the composite classified in full-image classification (both sensors).
DEFAULT_IMAGE = "Holambra_2022-12-15"

# Combined results table; the other tables are named after it (class_metrics.csv, ...).
RESULTS_FILENAME = "model_results.csv"

# Same band set of both sensors, for the paired S2 - PS differences.
SENSOR_PAIRS = {"4 bands": ("S2_1", "PS_1"), "all bands": ("S2_2", "PS_2")}


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

def _tuning_groups(df_train, tuning):
    if tuning and tuning.get("enabled", False):
        return polygon_groups(df_train)
    return None

def _get_imagery(paths, sensor, image=DEFAULT_IMAGE):
    """Composite of *sensor* for the ROI and reference date in *image* (e.g. "Holambra_2022-12-15"), or None."""
    matches = [path for path in get_image_lists(paths)[sensor] if path.stem == f"{sensor}_{image}"]
    return matches[0] if matches else None

# ---------------------------------------------------------------------------
# Individual experiment runners
# ---------------------------------------------------------------------------

def run_experiment_S2_4b(models, datasets, paths, apply_model_on_image,
                         tuning=None, random_state=DEFAULT_RANDOM_STATE, image=DEFAULT_IMAGE):
    X,  y  = datasets["df_s2train"][S2_BANDS_4], datasets["df_s2train"]["Class"]
    groups = _tuning_groups(datasets["df_s2train"], tuning)
    X2, y2 = datasets["df_s2val"][S2_BANDS_4],   datasets["df_s2val"]["Class"]

    imagery = _get_imagery(paths, "S2", image)
    return train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        condition_label="S2_1",
        apply_model_on_image=apply_model_on_image,
        tuning=tuning,
        random_state=random_state,
        groups=groups,
        models_path=paths["models_path"],
    )


def run_experiment_S2_Allb(models, datasets, paths, apply_model_on_image,
                         tuning=None, random_state=DEFAULT_RANDOM_STATE, image=DEFAULT_IMAGE):
    X,  y  = datasets["df_s2train"][S2_BANDS_ALL], datasets["df_s2train"]["Class"]
    groups = _tuning_groups(datasets["df_s2train"], tuning)
    X2, y2 = datasets["df_s2val"][S2_BANDS_ALL],   datasets["df_s2val"]["Class"]

    imagery = _get_imagery(paths, "S2", image)
    return train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        condition_label="S2_2",
        apply_model_on_image=apply_model_on_image,
        tuning=tuning,
        random_state=random_state,
        groups=groups,
        models_path=paths["models_path"],
    )


def run_experiment_PS_4b(models, datasets, paths, apply_model_on_image,
                         tuning=None, random_state=DEFAULT_RANDOM_STATE, image=DEFAULT_IMAGE):
    X,  y  = datasets["df_psstrain"][PS_BANDS_4], datasets["df_psstrain"]["Class"]
    groups = _tuning_groups(datasets["df_psstrain"], tuning)
    X2, y2 = datasets["df_pssval"][PS_BANDS_4],   datasets["df_pssval"]["Class"]

    imagery = _get_imagery(paths, "PS", image)
    return train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        condition_label="PS_1",
        apply_model_on_image=apply_model_on_image,
        tuning=tuning,
        random_state=random_state,
        groups=groups,
        models_path=paths["models_path"],
    )


def run_experiment_PS_Allb(models, datasets, paths, apply_model_on_image,
                         tuning=None, random_state=DEFAULT_RANDOM_STATE, image=DEFAULT_IMAGE):
    X,  y  = datasets["df_psstrain"][PS_BANDS_ALL], datasets["df_psstrain"]["Class"]
    groups = _tuning_groups(datasets["df_psstrain"], tuning)
    X2, y2 = datasets["df_pssval"][PS_BANDS_ALL],   datasets["df_pssval"]["Class"]

    imagery = _get_imagery(paths, "PS", image)
    return train_and_evaluate_model(
        X, X2, y, y2, models, imagery, paths["maps_path"],
        condition_label="PS_2",
        apply_model_on_image=apply_model_on_image,
        tuning=tuning,
        random_state=random_state,
        groups=groups,
        models_path=paths["models_path"],
    )


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

def run_all_experiments(
    selected_experiments: list,
    models: dict,
    datasets: dict,
    paths: dict,
    tuning: dict = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    uncertainty: dict = None,
    image: str = DEFAULT_IMAGE,
) -> pd.DataFrame:
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
    tuning : dict, optional
        Successive Halving settings; ``None`` keeps default hyperparameters.
    random_state : int
        Seed for the hyperparameter search and the bootstrap resamples.
    uncertainty : dict
        ``n_resamples`` and ``confidence`` of the class-stratified bootstrap of the
        validation points by polygon (see :mod:`src.uncertainty`).
    image : str
        ROI and reference date of the composite classified in full-image
        classification, as in its file name (e.g. ``"Holambra_2022-12-15"``).

    Returns
    -------
    pd.DataFrame
        Concatenated results from all experiments, with confidence intervals of OA,
        Kappa and Macro F1, also saved to ``results_path/model_results.csv``.  Saved next
        to it: ``class_metrics.csv`` (with intervals), ``confusion_matrices.csv``,
        ``validation_predictions.csv.gz`` (class predicted for every validation point) and
        ``sensor_differences.csv`` (paired S2 - PS differences of OA with intervals).
    """
    selected_results = []
    selected_class_metrics = []
    selected_confusion = []
    selected_predictions = []
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

        results_i, class_metrics_i, confusion_i, predictions_i = EXPERIMENT_QUEUE[name](
            models=models,
            datasets=datasets,
            paths=paths,
            apply_model_on_image=apply_flag,
            tuning=tuning,
            random_state=random_state,
            image=image,
        )
        selected_results.append(results_i)
        selected_class_metrics.append(class_metrics_i)
        selected_confusion.append(confusion_i)
        selected_predictions.append(predictions_i)

    combined = (
        pd.concat(selected_results, ignore_index=True)
        if selected_results
        else pd.DataFrame()
    )
    name = RESULTS_FILENAME
    class_metrics = pd.concat(selected_class_metrics, ignore_index=True) if selected_results else pd.DataFrame()
    if selected_results:
        predictions = pd.concat(selected_predictions, ignore_index=True)
        combined, class_metrics, differences = _add_intervals(
            combined, class_metrics, predictions, datasets["df_s2val"], uncertainty, random_state
        )
        predictions.to_csv(
            paths["results_path"] / name.replace("model_results", "validation_predictions").replace(".csv", ".csv.gz"),
            index=False,
        )
        differences.to_csv(paths["results_path"] / name.replace("model_results", "sensor_differences"), index=False)
    combined.drop(columns=INTERNAL_COLUMNS, errors="ignore").to_csv(paths["results_path"] / name, index=False)
    # Per-class metrics and confusion matrices on the validation set.
    if selected_results:
        class_metrics.to_csv(paths["results_path"] / name.replace("model_results", "class_metrics"), index=False)
        pd.concat(selected_confusion, ignore_index=True).to_csv(
            paths["results_path"] / name.replace("model_results", "confusion_matrices"), index=False
        )
    # Breakdown of the training time (search, final fit, inner-CV OA): internal, not part of results/.
    if not combined.empty:
        combined[["Imagery", "Models", *INTERNAL_COLUMNS]].to_csv(
            paths["scratch_path"] / name.replace("model_results", "time_breakdown"), index=False
        )
    return combined


def _add_intervals(results, class_metrics, predictions, df_val, settings, random_state):
    """
    Bootstrap confidence intervals of the validation metrics (class-stratified spatial
    blocks, the same resamples for every experiment and model) and the paired S2 - PS
    differences of OA (per classifier and band set, and their mean).
    """
    resamples = validation_resamples(df_val, settings, random_state)
    confidence = settings["confidence"]
    labels = np.unique(df_val["Class"])
    draws = {}
    for (imagery, model), group in predictions.groupby(["Imagery", "Models"], sort=False):
        group = group.set_index("Point").loc[df_val.index]  # validation-table order, as the resample indices
        counts = confusion_counts(group["True Class"], group["Predicted Class"], labels, resamples)
        draws[imagery, model] = metrics_from_confusion(counts)

    keys = list(zip(results["Imagery"], results["Models"]))
    for metric in ("OA", "Kappa", "Macro F1"):
        low, high = zip(*(interval(draws[k][metric], confidence) for k in keys))
        position = results.columns.get_loc(metric) + 1
        results.insert(position, f"{metric} CI Low", np.round(low, 6))
        results.insert(position + 1, f"{metric} CI High", np.round(high, 6))

    column = {label: i for i, label in enumerate(labels)}
    rows = list(zip(class_metrics["Imagery"], class_metrics["Models"], class_metrics["Class"]))
    for metric in ("Precision", "Recall", "F1"):
        bounds = {key: interval(d[metric], confidence) for key, d in draws.items()}
        position = class_metrics.columns.get_loc(metric) + 1
        class_metrics.insert(position, f"{metric} CI Low", [round(float(bounds[i, m][0][column[c]]), 6) for i, m, c in rows])
        class_metrics.insert(position + 1, f"{metric} CI High", [round(float(bounds[i, m][1][column[c]]), 6) for i, m, c in rows])

    oa = results.set_index(["Imagery", "Models"])["OA"]
    differences, pairs = [], []
    for bands, (s2, ps) in SENSOR_PAIRS.items():
        for model in results["Models"].unique():
            if (s2, model) in draws and (ps, model) in draws:
                pairs.append((s2, ps, model))
                low, high = interval(draws[s2, model]["OA"] - draws[ps, model]["OA"], confidence)
                differences.append({"Models": model, "Bands": bands, "S2 Imagery": s2, "PS Imagery": ps,
                                    "OA Difference (S2 - PS)": round(float(oa[s2, model] - oa[ps, model]), 6),
                                    "CI Low": round(float(low), 6), "CI High": round(float(high), 6)})
    # Mean over all pairs (classifiers and band sets), with the interval of the mean over the same resamples.
    if pairs:
        low, high = interval(np.mean([draws[s2, m]["OA"] - draws[ps, m]["OA"] for s2, ps, m in pairs], axis=0), confidence)
        differences.append({"Models": "Mean", "Bands": "4 bands and all bands",
                            "S2 Imagery": ", ".join(SENSOR_PAIRS[b][0] for b in SENSOR_PAIRS),
                            "PS Imagery": ", ".join(SENSOR_PAIRS[b][1] for b in SENSOR_PAIRS),
                            "OA Difference (S2 - PS)": round(float(np.mean([oa[s2, m] - oa[ps, m] for s2, ps, m in pairs])), 6),
                            "CI Low": round(float(low), 6), "CI High": round(float(high), 6)})
    return results, class_metrics, pd.DataFrame(differences)

