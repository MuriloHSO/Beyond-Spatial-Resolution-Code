"""
tuning.py
---------
Hyperparameter tuning with Successive Halving (``HalvingRandomSearchCV``).

Each candidate configuration is first evaluated on a small subsample of the
train-test set; only the best third (``factor = 3``) advances to the next round,
which uses three times more samples, until the survivors are evaluated on the
full train-test set.  Tuning uses only the train-test set (inner cross-validation),
so the validation set remains fully held out for the reported metrics.

Train-test points are only a few metres apart, so neighbouring points often fall
in the same pixel.  With random folds, near-duplicates end up on both sides of
a split and the search favours configurations that memorise them.  The inner
folds are therefore grouped by labelled polygon and season: all points of a
polygon on one reference date are in the same fold (``polygon_groups``).
"""

import time

import numpy as np

from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_RANDOM_STATE


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

SEARCH_SPACES = {
    "CART": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15, 20, 30, 40],
        "min_samples_split": randint(2, 50),
        "min_samples_leaf": randint(1, 30),
        "max_features": [None, "sqrt", "log2"],
    },
    "KNN": {
        "n_neighbors": randint(1, 50),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (200,), (100, 50), (100, 100)],
        "activation": ["relu", "tanh"],
        "alpha": loguniform(1e-6, 1e-1),
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "early_stopping": [True, False],
    },
    "RF": {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    },
    "SGDC": {
        "loss": ["hinge", "log_loss", "modified_huber"],
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": loguniform(1e-6, 1e-1),
        "l1_ratio": uniform(0, 1),
    },
    "LinearSVC": {
        "C": loguniform(1e-3, 1e2),
        "loss": ["hinge", "squared_hinge"],
    },
    "SVC_rbf": {
        "C": loguniform(1e-1, 1e3),
        "gamma": loguniform(1e-4, 1e1),
    },
}


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def polygon_groups(df):
    """
    Group of each point for the tuning folds and the bootstrap: its labelled polygon on
    its reference date (class, ``Polygon_ID``, ``Reference_Date``), or the point itself
    for "others" (no polygon). The same field on another date is an independent group.
    """
    polygon = df["Polygon_ID"]
    return np.where(
        polygon.isna(),
        "point-" + df.index.astype(str),
        df["Class"].astype(str) + "-" + polygon.astype("Int64").astype(str) + "-" + df["Reference_Date"].astype(str),
    )


def tune_model(model_name, model, X_train, y_train, groups, tuning: dict, random_state: int = DEFAULT_RANDOM_STATE):
    """
    Search the hyperparameters of *model* with Successive Halving.

    Scaling is done inside a pipeline so that every inner CV fold is
    standardised with statistics from its own training part only.

    Parameters
    ----------
    model_name : str
        Key in :data:`SEARCH_SPACES`.
    model : estimator
        Base (untrained) estimator; it is cloned, never modified.
    X_train, y_train : array-like
        Raw (unscaled) train-test data.
    groups : array-like
        Group of each train-test point: polygon and season, or the point itself (see :func:`polygon_groups`).
    tuning : dict
        ``n_candidates``, ``factor`` and ``cv_folds`` settings.
    random_state : int
        Seed for candidate sampling, subsampling and fold splitting.

    Returns
    -------
    (tuned_model, best_params, best_score, tuning_time)
        *tuned_model* is an unfitted clone of *model* with the best
        hyperparameters set.
    """
    space = {f"model__{k}": v for k, v in SEARCH_SPACES[model_name].items()}
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", clone(model))])

    search = HalvingRandomSearchCV(
        pipeline,
        space,
        n_candidates=tuning["n_candidates"],
        factor=tuning["factor"],
        resource="n_samples",
        min_resources="exhaust",
        cv=StratifiedGroupKFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=random_state),
        scoring="accuracy",
        refit=False,
        random_state=random_state,
        n_jobs=-1,
    )

    start_time = time.time()
    search.fit(X_train, y_train, groups=groups)
    tuning_time = time.time() - start_time

    best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    tuned_model = clone(model).set_params(**best_params)

    return tuned_model, best_params, float(search.best_score_), tuning_time
