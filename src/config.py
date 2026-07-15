"""
config.py
---------
Central configuration for model selection and experiment parameters.

Users can edit RANDOM_STATE and ENABLED_MODELS here, or override them
from the notebook.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


def build_models(random_state: int = 2025, enabled: list = None) -> dict:
    """
    Build the classifiers dictionary.

    Parameters
    ----------
    random_state : int
        Seed used for all classifiers that accept ``random_state``.
    enabled : list of str, optional
        Names of models to include.  If *None* (default) all seven models are
        returned.  Valid names: 'CART', 'KNN', 'MLP', 'RF', 'SGD',
        'SVM_linear', 'SVM_rbf'.

    Returns
    -------
    dict[str, estimator]
    """
    all_models = {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state),
        "SGD": SGDClassifier(random_state=random_state),
        "SVM_linear": LinearSVC(random_state=random_state),
        "SVM_rbf": SVC(random_state=random_state),
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
