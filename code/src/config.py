"""
config.py
---------
Central configuration for model selection and experiment parameters.

The random seed is set in config.toml (``random_state``); DEFAULT_RANDOM_STATE is the
fallback used everywhere when no value is given.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Single seed for every source of randomness: classifiers, CV splits and the hyperparameter search.
DEFAULT_RANDOM_STATE = 2025

# Names of the three classes coded in the "Class" column of the datasets (used in figure labels),
# as assigned by src/generate_data_points.py (CLASSES).
CLASS_NAMES = {0: "PMF", 1: "PGH", 2: "others"}


def build_models(random_state: int = DEFAULT_RANDOM_STATE, enabled: list = None) -> dict:
    """
    Build the classifiers dictionary.

    Parameters
    ----------
    random_state : int
        Seed used for all classifiers that accept ``random_state``.
    enabled : list of str, optional
        Names of models to include.  If *None* (default) all seven models are
        returned.  Valid names: 'CART', 'KNN', 'MLP', 'RF', 'SGDC',
        'LinearSVC', 'SVC_rbf'.

    Returns
    -------
    dict[str, estimator]
    """
    # max_iter above the scikit-learn defaults (MLP 200, LinearSVC 1000): with these limits MLP converges
    # and LinearSVC results become stable across seeds (see README, "Hyperparameter tuning").
    all_models = {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=random_state, max_iter=1000),
        "RF": RandomForestClassifier(random_state=random_state),
        "SGDC": SGDClassifier(random_state=random_state),
        "LinearSVC": LinearSVC(random_state=random_state, max_iter=10000),
        "SVC_rbf": SVC(random_state=random_state),
    }

    if enabled is None:
        return all_models

    unknown = [m for m in enabled if m not in all_models]
    if unknown:
        raise ValueError(
            f"Unknown model name(s): {unknown}. "
            f"Valid options are: {list(all_models.keys())}"
        )

    return {name: all_models[name] for name in enabled}
