"""
training.py
-----------
Core model training and evaluation logic.
"""

import gc
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import sklearn
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import colors
from threadpoolctl import ThreadpoolController

from .config import DEFAULT_RANDOM_STATE
from .tuning import tune_model

# Breakdown of the training time, kept out of results/ (written to scratch/ by the dispatcher).
INTERNAL_COLUMNS = ["Tuning Time (s)", "Final Fit Time (s)", "Tuning CV OA"]

# Red, green and blue bands at t0 of each sensor's composite, for the image preview.
RGB_BANDS = {"S2": ["B4_0", "B3_0", "B2_0"], "PS": ["B6_0", "B4_0", "B2_0"]}

# Pixels per block in full-image classification. Every pixel is classified independently, so the
# image is split into blocks predicted in parallel, one block per CPU thread at a time, with one
# thread inside each block, the same for every model. Predicting the whole PS image in one call
# would also need more memory than most computers have (MLP with 200 hidden units: ~60 GB).
PREDICT_BLOCK_PIXELS = 100_000


def _predict_block(model, block, controller):
    # OpenMP thread limits are per thread (KNN), so they are set in each worker thread.
    with controller.limit(limits=1, user_api="openmp"):
        return model.predict(block)


def predict_image(model, features):
    """Predict the classes of *features* (pixels × bands) in parallel blocks of PREDICT_BLOCK_PIXELS rows."""
    n_jobs = model.get_params().get("n_jobs")
    if n_jobs is not None:
        model.set_params(n_jobs=1)
    controller = ThreadpoolController()
    try:
        with controller.limit(limits=1, user_api="blas"):
            blocks = Parallel(n_jobs=-1, backend="threading")(
                delayed(_predict_block)(model, features[start:start + PREDICT_BLOCK_PIXELS], controller)
                for start in range(0, len(features), PREDICT_BLOCK_PIXELS)
            )
    finally:
        if n_jobs is not None:
            model.set_params(n_jobs=n_jobs)
    return np.concatenate(blocks)


def train_and_evaluate_model(
    X_train,
    X_val,
    y_train,
    y_val,
    models: dict,
    imagery,
    results_path,
    apply_model_on_image: bool = True,
    condition_label=None,
    tuning: dict = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    groups=None,
    models_path=None,
    save_fitted: bool = True,
) -> tuple:
    """
    Train every model in *models*, evaluate on the validation set and (optionally)
    classify the full image.

    Parameters
    ----------
    X_train, X_val : array-like
        Train-test and validation features.
    y_train, y_val : array-like
        Train-test and validation classes.
    models : dict[str, estimator]
    imagery : str or Path
        Path to the GeoTIFF used for full-image classification.  Ignored when
        *apply_model_on_image* is ``False``.
    results_path : Path
        Where PNG / TIFF outputs are written.
    apply_model_on_image : bool
        If ``False`` the image is never read; classification time is reported
        as NaN.
    condition_label : str, optional
        Overrides the auto-generated experiment name.
    tuning : dict, optional
        Successive Halving settings (see :func:`src.tuning.tune_model`).  If
        ``None`` or ``tuning["enabled"]`` is ``False``, the default
        hyperparameters are used.
    random_state : int
        Seed for the hyperparameter search.
    groups : array-like, optional
        Spatial block of each train-test point, used to split the tuning folds.
    models_path : Path, optional
        Where each fitted model is saved: ``<imagery>_<model>.joblib`` (scaler +
        model, lzma-compressed) and ``<imagery>_<model>.json`` (hyperparameters,
        features, scaler statistics, versions). Not saved when ``None``.
    save_fitted : bool
        If ``False``, only the JSON record is saved (no ``.joblib``).

    Returns
    -------
    (params, results_df, class_metrics_df, confusion_df, predictions_df)
        ``class_metrics_df`` has the per-class precision, recall, F1 and support on the
        validation set; ``confusion_df`` has the confusion matrix in long format
        (true class, predicted class, count); ``predictions_df`` has the predicted
        class of every validation point (``Point`` = row of the validation table).
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    filename = Path(imagery).stem if imagery is not None else "dataset"
    if condition_label is not None:
        name = str(condition_label)
    else:
        sat_prefix = (
            "S2"
            if filename.upper().startswith("S2")
            else ("PS" if filename.upper().startswith("PS") else "IMG")
        )
        condition_idx = 1 if X_train.shape[1] <= 12 else 2
        name = f"{sat_prefix}_{condition_idx}"

    image_features = None
    meta = None

    if apply_model_on_image:
        if imagery is None:
            print(f"    [!]  No imagery file found for '{name}' - image classification skipped.")
            apply_model_on_image = False
        else:
            # Load image only when explicit image classification is requested.
            # Composite bands are named like the dataset columns (e.g. B8_1), so the
            # model's features are read by name, in the training order.
            with rasterio.open(imagery) as src:
                bands = list(src.descriptions)
                image = src.read([bands.index(column) + 1 for column in X_train.columns])
                rgb = src.read([bands.index(band) + 1 for band in RGB_BANDS[filename[:2].upper()]])
                meta = src.meta

            img = np.clip(rgb / 10000 * 255 * 1.4, 0, 255).astype(np.uint8)

            # Save original image (true colour at t0)
            plt.figure(figsize=(16, 16))
            plt.imshow(np.dstack(img))
            plt.axis("off")
            plt.savefig(
                results_path / f"original_{filename}.png",
                bbox_inches="tight",
                dpi=600,
                pad_inches=0,
            )
            plt.close()

            # Reshape image
            image_reshaped = image.reshape(image.shape[0], -1).T

            # Imputation: pixels without a clear observation in a window are 0 (nodata)
            imputer = SimpleImputer(missing_values=0, strategy="mean")
            image_imputed = imputer.fit_transform(image_reshaped)

            # Normalize only necessary part
            image_features = scaler.transform(image_imputed)

            # Free memory
            del image, img, image_reshaped, image_imputed
            gc.collect()
    # Train and evaluate models
    params = []
    results = []
    class_rows = []
    confusion_rows = []
    prediction_frames = []
    labels = np.unique(y_train)
    n_models = len(models)
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        # Hyperparameter tuning (train-test set only)
        best_params, tuning_score, tuning_time = {}, np.nan, np.nan
        if tuning and tuning.get("enabled", False):
            model, best_params, tuning_score, tuning_time = tune_model(
                model_name, model, X_train, y_train, groups, tuning, random_state
            )

        # Timed fits and predictions use all CPU threads where the estimator supports it
        # (after the search, which already runs in parallel).
        if "n_jobs" in model.get_params():
            model.set_params(n_jobs=-1)

        # Training
        training_times = []
        for _ in range(3):
            start_time = time.time()
            model.fit(X_train_norm, y_train)
            training_times.append(time.time() - start_time)
        training_time = float(np.mean(training_times))
        training_time_std = round(float(np.std(training_times)), 6)

        validation_times = []
        for _ in range(3):
            start_time = time.time()
            y_pred = model.predict(X_val_norm)
            validation_times.append(time.time() - start_time)
        validation_time = float(np.mean(validation_times))
        validation_time_std = round(float(np.std(validation_times)), 6)

        classification_time = np.nan
        classification_time_std = np.nan
        if apply_model_on_image:
            classification_times = []
            for _ in range(3):
                start_time = time.time()
                prediction = predict_image(model, image_features)
                classification_times.append(time.time() - start_time)
            classification_time = float(np.mean(classification_times))
            classification_time_std = round(float(np.std(classification_times)), 6)

        if models_path is not None:
            _save_model(models_path, name, model_name, scaler, model, list(X_train.columns),
                        best_params, random_state, len(y_train), save_fitted)
        prediction_frames.append(pd.DataFrame({
            "Imagery": name, "Models": model_name, "Point": np.asarray(y_val.index),
            "True Class": np.asarray(y_val), "Predicted Class": y_pred,
        }))

        # Metrics (confidence intervals are added by the dispatcher, src.experiments)
        accuracy = round(accuracy_score(y_val, y_pred), 6)
        kappa = round(cohen_kappa_score(y_val, y_pred), 6)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, y_pred, labels=labels, zero_division=0
        )
        macro_f1 = round(float(np.mean(f1)), 6)
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            class_rows.append([name, model_name, label, round(p, 6), round(r, 6), round(f, 6), int(s)])
        cm = confusion_matrix(y_val, y_pred, labels=labels)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                confusion_rows.append([name, model_name, true_label, pred_label, int(cm[i, j])])
        bullet  = "\\-" if model_idx == n_models else "+-"
        img_str = f"{classification_time:.3f}s" if apply_model_on_image else "-"
        tune_str = f"  tune {tuning_time:.1f}s" if best_params else ""
        print(
            f"  {bullet} [{model_idx}/{n_models}] {model_name:<12}"
            f"{tune_str}"
            f"  train {training_time:.3f}s"
            f"  valid {validation_time:.3f}s"
            f"  classif {img_str}  [OK]"
        )
        if best_params:
            print(f"  {'|' if model_idx < n_models else ' '}      best: {_format_params(best_params)}")

        params.append(model.get_params(deep=True))
        results.append(
            [
                name,
                model_name,
                accuracy,
                kappa,
                macro_f1,
                # TT = time to obtain a trained model: hyperparameter search (once) + final fit (mean of 3)
                round(training_time + (tuning_time if best_params else 0.0), 6),
                training_time_std,
                round(validation_time, 6),
                validation_time_std,
                round(classification_time, 6) if apply_model_on_image else np.nan,
                classification_time_std,
                _format_params(best_params, full_precision=True),
                round(tuning_time, 6),
                round(training_time, 6),
                round(tuning_score, 6),
            ]
        )

        if apply_model_on_image:
            # Reshape prediction and save
            prediction_reshaped = prediction.reshape(meta["height"], meta["width"])

            plt.figure(figsize=(16, 16))
            plt.imshow(prediction_reshaped, cmap=colors.ListedColormap(['red', 'green', 'white']))
            plt.axis('off')
            plt.savefig(
                results_path / "PNG" / f"{model_name}_result_{filename}_{X_train.shape[1]}bands.png",
                bbox_inches="tight",
                dpi=600,
                pad_inches=0,
            )
            plt.close()

            # Save as TIFF
            # No nodata value: every pixel is classified, and 0 is a class code (PMF).
            meta.update(count=1, nodata=None)
            with rasterio.open(
                results_path / "TIFF" / f"{model_name}_result_{filename}_{X_train.shape[1]}bands.tif",
                "w",
                **meta,
            ) as dest:
                dest.write(prediction_reshaped, 1)

    # Create results DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "Imagery",
            "Models",
            "OA",
            "Kappa",
            "Macro F1",
            "Training Time (s)",
            "Training Time Std (s)",
            "Validation Time (s)",
            "Validation Time Std (s)",
            "Classification Time (s)",
            "Classification Time Std (s)",
            "Best Params",
            *INTERNAL_COLUMNS,
        ],
    )

    class_metrics_df = pd.DataFrame(
        class_rows, columns=["Imagery", "Models", "Class", "Precision", "Recall", "F1", "Support"]
    )
    confusion_df = pd.DataFrame(
        confusion_rows, columns=["Imagery", "Models", "True Class", "Predicted Class", "Count"]
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return params, results_df, class_metrics_df, confusion_df, predictions_df


def _save_model(models_path, imagery, model_name, scaler, model, features, best_params,
                random_state, n_train, save_fitted=True) -> None:
    """Save the fitted scaler + model (joblib, lzma; unless *save_fitted* is False) and a JSON record that allows an exact refit."""
    stem = Path(models_path) / f"{imagery}_{model_name}"
    if save_fitted:
        joblib.dump(Pipeline([("scaler", scaler), ("model", model)]), f"{stem}.joblib", compress=("lzma", 6))
    record = {
        "imagery": imagery,
        "model": model_name,
        "estimator": type(model).__name__,
        "features": features,
        "hyperparameters": {k: _json_value(v) for k, v in sorted(model.get_params().items())},
        "tuned_hyperparameters": {k: _json_value(v) for k, v in sorted(best_params.items())},
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "random_state": random_state,
        "train_test_points": int(n_train),
        "scikit_learn": sklearn.__version__,
    }
    with open(f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return repr(value)


def _format_params(params: dict, full_precision: bool = False) -> str:
    """``key=value`` string of hyperparameters.

    Floats are rounded to 4 significant digits for display (log), or written at full precision
    (shortest repr that round-trips exactly) for the results CSV, so models can be refitted exactly.
    """
    parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, (float, np.floating)):
            value = repr(float(value)) if full_precision else f"{value:.4g}"
        elif isinstance(value, np.integer):
            value = int(value)
        parts.append(f"{key}={value}")
    return ", ".join(parts)
