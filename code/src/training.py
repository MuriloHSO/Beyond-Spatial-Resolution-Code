"""
training.py
-----------
Core model training and evaluation logic.
"""

import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from matplotlib import colors


def train_and_evaluate_model(
    X_train,
    X_test,
    y_train,
    y_test,
    models: dict,
    imagery,
    results_path,
    apply_model_on_image: bool = True,
    condition_label=None,
) -> tuple:
    """
    Train every model in *models*, evaluate on the validation set and (optionally)
    classify the full image.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like
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

    Returns
    -------
    (params, results_df)
    """
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # K-Fold setup for robust variability estimates
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
    kappa_scorer = make_scorer(cohen_kappa_score)

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
            # Load image only when explicit image classification is requested
            with rasterio.open(imagery) as src:
                image = src.read()
                meta = src.meta

            img = (image / 10000 * 255 * 1.4).astype(np.uint8)

            # Save original image
            plt.figure(figsize=(16, 16))
            try:
                # Use bands 2, 1, 0 (RGB) for visualization
                plt.imshow(np.dstack((img[2], img[1], img[0])))
            except Exception:
                # Fallback if fewer bands are available
                plt.imshow(img[0])
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

            # Band selection based on satellite type
            if filename[:2] == "S2" and X_train.shape[1] == 12:
                image_reshaped = image_reshaped[:, [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]]
            elif filename[:2] == "PS" and X_train.shape[1] == 12:
                image_reshaped = image_reshaped[:, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]]

            # Imputation
            imputer = SimpleImputer(strategy="mean")
            image_imputed = imputer.fit_transform(image_reshaped)

            # Normalize only necessary part
            image_features = scaler.transform(image_imputed)

            # Free memory
            del image, img, image_reshaped, image_imputed
            gc.collect()
    # Train and evaluate models
    params = []
    results = []
    n_models = len(models)
    for model_idx, (model_name, model) in enumerate(models.items(), 1):
        # K-Fold variability on training data
        cv_acc_scores = cross_val_score(
            model, X_train_norm, y_train, cv=kfold, scoring="accuracy", n_jobs=-1
        )
        accuracy_std = round(float(np.std(cv_acc_scores)), 6)

        cv_kappa_scores = cross_val_score(
            model, X_train_norm, y_train, cv=kfold, scoring=kappa_scorer, n_jobs=-1
        )
        kappa_std = round(float(np.std(cv_kappa_scores)), 6)

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
            y_pred = model.predict(X_test_norm)
            validation_times.append(time.time() - start_time)
        validation_time = float(np.mean(validation_times))
        validation_time_std = round(float(np.std(validation_times)), 6)

        classification_time = np.nan
        classification_time_std = np.nan
        if apply_model_on_image:
            classification_times = []
            for _ in range(3):
                start_time = time.time()
                prediction = model.predict(image_features)
                classification_times.append(time.time() - start_time)
            classification_time = float(np.mean(classification_times))
            classification_time_std = round(float(np.std(classification_times)), 6)

        # Metrics
        accuracy = round(accuracy_score(y_test, y_pred), 6)
        kappa = round(cohen_kappa_score(y_test, y_pred), 6)
        bullet  = "\\-" if model_idx == n_models else "+-"
        img_str = f"{classification_time:.3f}s" if apply_model_on_image else "-"
        print(
            f"  {bullet} [{model_idx}/{n_models}] {model_name:<12}"
            f"  train {training_time:.3f}s"
            f"  valid {validation_time:.3f}s"
            f"  classif {img_str}  [OK]"
        )

        params.append(model.get_params(deep=True))
        results.append(
            [
                name,
                model_name,
                accuracy,
                accuracy_std,
                kappa,
                kappa_std,
                round(training_time, 6),
                training_time_std,
                round(validation_time, 6),
                validation_time_std,
                round(classification_time, 6) if apply_model_on_image else np.nan,
                classification_time_std,
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
            meta.update(count=1)
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
            "OA Std (KFold)",
            "Kappa",
            "Kappa Std (KFold)",
            "Training Time (s)",
            "Training Time Std (s)",
            "Validation Time (s)",
            "Validation Time Std (s)",
            "Classification Time (s)",
            "Classification Time Std (s)",
        ],
    )

    return params, results_df
