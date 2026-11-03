"""
uncertainty.py
--------------
Confidence intervals of the validation metrics by a class-stratified bootstrap
of polygons.

Points of the same labelled polygon on the same date are a few metres apart and
not independent, so they are resampled together: for each class, its groups
(a polygon on one reference date, or a single "others" point; see
src.tuning.polygon_groups) are drawn with replacement. Drawing per class keeps
the class mix close to the balanced design.

The same resamples are used for every experiment and model (the validation
points are the same for both sensors), so differences between them, such as
S2 - PS, get paired intervals.
"""

import numpy as np

from .tuning import polygon_groups


def class_group_resamples(y, groups, n_resamples: int, random_state: int) -> list:
    """Index arrays of *n_resamples* class-stratified bootstrap resamples of *groups*."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    rng = np.random.default_rng(random_state)
    by_class = []
    for label in np.unique(y):
        in_class = np.flatnonzero(y == label)
        members = groups[in_class]
        by_class.append([in_class[members == g] for g in np.unique(members)])
    resamples = []
    for _ in range(n_resamples):
        parts = [np.concatenate([groups_[i] for i in rng.integers(0, len(groups_), len(groups_))]) for groups_ in by_class]
        resamples.append(np.concatenate(parts))
    return resamples


def validation_resamples(df_val, settings: dict, random_state: int) -> list:
    """Resamples of a validation table (Class, Polygon_ID, Reference_Date) with the [uncertainty] settings."""
    return class_group_resamples(df_val["Class"], polygon_groups(df_val), settings["n_resamples"], random_state)


def confusion_counts(y_true, y_pred, labels, resamples) -> np.ndarray:
    """Confusion matrices (resample x true x predicted) of every resample."""
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    codes = np.array([index[t] for t in y_true]) * k + np.array([index[p] for p in y_pred])
    return np.stack([np.bincount(codes[r], minlength=k * k).reshape(k, k) for r in resamples])


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """OA, kappa, macro F1 and per-class precision, recall and F1 of confusion matrices (..., k, k)."""
    cm = cm.astype(float)
    n = cm.sum(axis=(-2, -1))
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    true_totals, pred_totals = cm.sum(axis=-1), cm.sum(axis=-2)
    oa = diag.sum(axis=-1) / n
    expected = (true_totals * pred_totals).sum(axis=-1) / n**2
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = (oa - expected) / (1 - expected)
        precision = np.where(pred_totals > 0, diag / pred_totals, 0.0)
        recall = np.where(true_totals > 0, diag / true_totals, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"OA": oa, "Kappa": kappa, "Macro F1": f1.mean(axis=-1),
            "Precision": precision, "Recall": recall, "F1": f1}


def interval(values, confidence: float) -> tuple:
    """Percentile interval of bootstrap *values* (first axis = resamples)."""
    alpha = (1 - confidence) / 2
    return np.quantile(values, alpha, axis=0), np.quantile(values, 1 - alpha, axis=0)
