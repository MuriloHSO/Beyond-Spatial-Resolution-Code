"""
swir_ablation.py
----------------
SM Section B.3 and Table B.2: contribution of the S2 SWIR bands (B11, B12).

Two more S2 image sets, each in the three time windows, are tuned, trained and validated like the experiments:
    S2_4S  the four equivalent bands plus SWIR (B2, B3, B4, B8, B11, B12)
    S2_8   all bands without SWIR (B2-B8A)
They are compared with the four- and ten-band S2 sets and the PS sets of the same run (read from
validation_predictions.csv.gz), by paired differences of OA, macro F1 and per-class F1 over the same bootstrap
resamples of the validation points as the other intervals (src.uncertainty).
"""

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, DEFAULT_RANDOM_STATE
from .experiments import S2_BANDS_ALL, _tuning_groups
from .training import train_and_evaluate_model
from .uncertainty import confusion_counts, interval, metrics_from_confusion, validation_resamples

IMAGE_SETS = {
    "S2_4S": ["B2", "B3", "B4", "B8", "B11", "B12"],
    "S2_8": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"],
}
# Paired differences (first - second): SWIR added to four bands and to the other bands, red-edge bands added to
# four bands, and S2 - PS with and without SWIR.
COMPARISONS = [("S2_4S", "S2_1"), ("S2_2", "S2_8"), ("S2_8", "S2_1"), ("S2_2", "S2_1"),
               ("S2_8", "PS_2"), ("S2_1", "PS_1"), ("S2_2", "PS_2")]
REFERENCE = ["S2_1", "S2_2", "PS_1", "PS_2"]  # image sets of the run used in the comparisons
FOLDER = "swir_ablation"  # in results/ (tables) and results/Models/ (JSON records of the models)
METRICS = "swir_ablation.csv"
CLASS_METRICS = "swir_ablation_class_metrics.csv"
DIFFERENCES = "swir_ablation_differences.csv"


def run_swir_ablation(models, datasets, paths, tuning=None, random_state=DEFAULT_RANDOM_STATE, uncertainty=None):
    """
    Run the two S2 image sets and write, to ``results/swir_ablation/``:
    ``swir_ablation.csv`` (OA, kappa, macro F1 with intervals, and the selected hyperparameters),
    ``swir_ablation_class_metrics.csv`` (per-class precision, recall and F1 with intervals) and
    ``swir_ablation_differences.csv`` (paired differences of OA, macro F1 and per-class F1 per classifier and
    their mean over classifiers, ``Mean``, with intervals); and the JSON record of every model to
    ``results/Models/swir_ablation/``. Needs ``validation_predictions.csv.gz`` of the run with the four experiments.
    """
    out_path = paths["results_path"] / FOLDER
    models_path = paths["models_path"] / FOLDER
    out_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(paths["results_path"] / "validation_predictions.csv.gz")
    reference = reference[reference["Imagery"].isin(REFERENCE)]
    missing = sorted(set(REFERENCE) - set(reference["Imagery"]))
    if missing:
        raise ValueError(f"needs the predictions of {', '.join(missing)} (run all four experiments)")

    train, val = datasets["df_s2train"], datasets["df_s2val"]
    groups = _tuning_groups(train, tuning)
    results, class_rows, predictions = [], [], [reference]
    for name, bands in IMAGE_SETS.items():
        features = [c for c in S2_BANDS_ALL if c.rsplit("_", 1)[0] in bands]
        print(f"\n  + {name} ({', '.join(bands)})")
        res, cls, _, pred = train_and_evaluate_model(
            train[features], val[features], train["Class"], val["Class"], models, None, paths["maps_path"],
            apply_model_on_image=False, condition_label=name, tuning=tuning, random_state=random_state,
            groups=groups, models_path=models_path, save_fitted=False,
        )
        results.append(res)
        class_rows.append(cls)
        predictions.append(pred)
    predictions = pd.concat(predictions, ignore_index=True)

    resamples = validation_resamples(val, uncertainty, random_state)
    labels = np.unique(val["Class"])
    draws = {}
    for key, group in predictions.groupby(["Imagery", "Models"], sort=False):
        group = group.set_index("Point").loc[val.index]  # validation-table order, as the resample indices
        draws[key] = metrics_from_confusion(confusion_counts(group["True Class"], group["Predicted Class"],
                                                             labels, resamples))
    confidence = uncertainty["confidence"]

    results = pd.concat(results, ignore_index=True)
    out = results[["Imagery", "Models", "OA", "Kappa", "Macro F1", "Best Params"]].copy()
    for metric in ("OA", "Kappa", "Macro F1"):
        low, high = zip(*(interval(draws[k][metric], confidence) for k in zip(out["Imagery"], out["Models"])))
        position = out.columns.get_loc(metric) + 1
        out.insert(position, f"{metric} CI Low", np.round(low, 6))
        out.insert(position + 1, f"{metric} CI High", np.round(high, 6))
    out.to_csv(out_path / METRICS, index=False)

    class_metrics = pd.concat(class_rows, ignore_index=True)
    column = {label: i for i, label in enumerate(labels)}
    rows = list(zip(class_metrics["Imagery"], class_metrics["Models"], class_metrics["Class"]))
    for metric in ("Precision", "Recall", "F1"):
        bounds = {key: interval(d[metric], confidence) for key, d in draws.items()}
        position = class_metrics.columns.get_loc(metric) + 1
        class_metrics.insert(position, f"{metric} CI Low", [round(float(bounds[i, m][0][column[c]]), 6) for i, m, c in rows])
        class_metrics.insert(position + 1, f"{metric} CI High", [round(float(bounds[i, m][1][column[c]]), 6) for i, m, c in rows])
    class_metrics.to_csv(out_path / CLASS_METRICS, index=False)

    # Point estimates from all validation points; intervals from the resamples
    full = {key: metrics_from_confusion(confusion_counts(g["True Class"], g["Predicted Class"], labels,
                                                         [np.arange(len(g))]))
            for key, g in predictions.groupby(["Imagery", "Models"], sort=False)}
    model_names = list(results["Models"].unique())
    values = [("OA", None), ("Macro F1", None)] + [("F1", i) for i in range(len(labels))]
    rows = []
    for first, second in COMPARISONS:
        for model in model_names + ["Mean"]:
            pairs = [(m, m) for m in (model_names if model == "Mean" else [model])]
            row = {"First": first, "Second": second, "Models": model}
            for metric, i in values:
                pick = (lambda d: d[metric]) if i is None else (lambda d: d[metric][..., i])
                label = metric if i is None else f"F1 {CLASS_NAMES.get(labels[i], labels[i])}"
                diff = np.mean([pick(draws[first, a]) - pick(draws[second, b]) for a, b in pairs], axis=0)
                estimate = np.mean([pick(full[first, a])[0] - pick(full[second, b])[0] for a, b in pairs])
                low, high = interval(diff, confidence)
                row.update({f"{label} Difference": round(float(estimate), 6),
                            f"{label} CI Low": round(float(low), 6), f"{label} CI High": round(float(high), 6)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path / DIFFERENCES, index=False)
    return out
