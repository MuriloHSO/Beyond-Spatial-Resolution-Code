"""
run.py
------
Main entry point for Code Ocean (and local execution without Jupyter).

Configuration is read from config.toml in the same directory as this file.
Edit that file to change models, experiments, and other settings, then run:

    python run.py

Command-line arguments can still override any config.toml value:

    python run.py --help

Options
-------
    --config PATH               Path to a TOML config file
                                (default: config.toml next to this script)
    --random-state INT          Random seed for classifiers
    --models MODEL [MODEL ...]  Models to run; omit to use config.toml value.
                                Valid: CART KNN MLP RF SGD SVM_linear SVM_rbf
    --experiments EXP [EXP ...] Experiments to run; omit to use config.toml.
                                Valid: S2_4b S2_Allb PS_4b PS_Allb
    --apply-on-image            Force full-image classification for every
                                experiment (overrides config.toml per-entry flag)
    --no-image                  Disable full-image classification for every
                                experiment (overrides config.toml per-entry flag)
    --skip-plots                Do not generate figures after the experiments.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Check if required packages are installed, and run pip install if not."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pkg_resources
    except ImportError:
        # If pkg_resources is missing, just try running pip install directly
        print("Checking requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        return

    # Parse requirements.txt
    with open(req_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    try:
        pkg_resources.require(requirements)
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
        print(f"Missing or outdated dependency: {e.req}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("Dependencies installed successfully.\n")

check_requirements()

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for Code Ocean
from matplotlib import rc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# TOML loading (tomllib is built-in for Python 3.11+; fall back to tomli)
# ---------------------------------------------------------------------------

try:
    import tomllib                          # Python 3.11+
except ModuleNotFoundError:
    try:
        import tomli as tomllib             # pip install tomli
    except ModuleNotFoundError:
        tomllib = None


def load_toml_config(path: Path) -> dict:
    """Load a TOML config file. Returns an empty dict if TOML is unavailable."""
    if tomllib is None:
        print(
            f"WARNING: Could not import 'tomllib' or 'tomli'. "
            f"Config file '{path}' will be ignored.\n"
            f"  On Python <3.11 run: pip install tomli"
        )
        return {}
    if not path.exists():
        print(f"WARNING: Config file not found at '{path}'. Using defaults.")
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

ALL_MODELS = ["CART", "KNN", "MLP", "RF", "SGD", "SVM_linear", "SVM_rbf"]
ALL_EXPERIMENTS = ["S2_4b", "S2_Allb", "PS_4b", "PS_Allb"]

# Fallback defaults (used when config.toml is missing and no CLI flag is given)
_FALLBACK_RANDOM_STATE = 2025
_FALLBACK_MODELS: list = []          # empty → all models
_FALLBACK_EXPERIMENTS = [
    {"name": "S2_4b",   "apply_model_on_image": True},
    {"name": "S2_Allb", "apply_model_on_image": True},
    {"name": "PS_4b",   "apply_model_on_image": True},
    {"name": "PS_Allb", "apply_model_on_image": True},
]

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Beyond Spatial Resolution — training and evaluation script."
    )
    parser.add_argument(
        "--config", type=Path,
        default=None,
        help="Path to a TOML config file (default: config.toml next to run.py).",
    )
    parser.add_argument(
        "--random-state", type=int, default=None,
        help="Random seed for all classifiers.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=None,
        help="Subset of models to run. Omit to use config.toml value.",
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=ALL_EXPERIMENTS, default=None,
        help="Subset of experiments to run. Omit to use config.toml value.",
    )

    # Mutually exclusive image-classification overrides
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument(
        "--apply-on-image", action="store_true", default=False,
        help="Force full-image classification for every experiment.",
    )
    img_group.add_argument(
        "--no-image", action="store_true", default=False,
        help="Disable full-image classification for every experiment.",
    )

    parser.add_argument(
        "--skip-plots", action="store_true", default=False,
        help="Skip figure generation after the experiments.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config resolution  (config.toml values, overridden by CLI args)
# ---------------------------------------------------------------------------


def resolve_config(args) -> dict:
    """Merge config.toml values with CLI overrides and return a final config dict."""
    # Determine which TOML file to load
    script_dir = Path(__file__).resolve().parent
    config_path = args.config if args.config is not None else script_dir / "config.toml"
    toml = load_toml_config(config_path)

    # --- random_state ---
    random_state = (
        args.random_state                           # CLI wins
        if args.random_state is not None
        else toml.get("random_state", _FALLBACK_RANDOM_STATE)
    )

    # --- enabled models ---
    if args.models is not None:
        # CLI wins
        enabled_models = args.models if args.models else None
    else:
        toml_models = toml.get("models", _FALLBACK_MODELS)
        enabled_models = toml_models if toml_models else None   # [] → None (all)

    # --- experiments ---
    # apply_model_on_image is now a single global boolean in config.toml
    apply_flag = bool(toml.get("apply_model_on_image", True))

    if args.experiments is not None:
        # CLI wins for experiment selection; use the global TOML boolean for image flag
        selected_experiments = [
            {"name": name, "apply_model_on_image": apply_flag}
            for name in args.experiments
        ]
    else:
        exp_names = toml.get("experiments", [e["name"] for e in _FALLBACK_EXPERIMENTS])
        selected_experiments = [
            {"name": name, "apply_model_on_image": apply_flag}
            for name in exp_names
        ]

    # --- global image-classification override ---
    if args.apply_on_image:
        for exp in selected_experiments:
            exp["apply_model_on_image"] = True
    elif args.no_image:
        for exp in selected_experiments:
            exp["apply_model_on_image"] = False

    # --- skip_plots ---
    skip_plots = args.skip_plots or toml.get("skip_plots", False)

    return {
        "random_state": random_state,
        "enabled_models": enabled_models,
        "selected_experiments": selected_experiments,
        "skip_plots": skip_plots,
    }


# ---------------------------------------------------------------------------
# Path resolution (Code Ocean vs. local)
# ---------------------------------------------------------------------------


def resolve_base_path():
    """
    Return the repository root, accounting for Code Ocean's layout.

    Code Ocean mounts:
      /code    — the capsule code directory (repo root)
      /data    — input datasets and imagery
      /results — output directory

    Local layout mirrors Code Ocean conventions:
      <repo>/data/    — CSVs and Imagery/ subfolder
      <repo>/results/ — output directory
    """
    code_ocean_data    = Path("/data")
    code_ocean_results = Path("/results")

    if code_ocean_data.exists() and code_ocean_results.exists():
        # Running inside Code Ocean
        return Path("/code"), code_ocean_data, code_ocean_results

    # Local execution: mirror Code Ocean conventions relative to this file's parent
    base = Path(__file__).resolve().parent
    return base, base / "data", base / "results"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Log-file tee
# ---------------------------------------------------------------------------


class _LogTee:
    """
    Write every message to both *wrapped* (the original stream) and a shared
    *log_file* simultaneously.  Assigned to sys.stdout and sys.stderr so that
    every print() and traceback is captured without changing the rest of the
    code.
    """

    def __init__(self, wrapped, log_file):
        self._wrapped  = wrapped
        self._log_file = log_file

    def write(self, msg):
        self._wrapped.write(msg)
        self._log_file.write(msg)
        return len(msg)

    def flush(self):
        self._wrapped.flush()
        self._log_file.flush()

    def __getattr__(self, name):          # forward fileno(), encoding, etc.
        return getattr(self._wrapped, name)


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------


def _print_banner():
    """Print a welcome banner to stdout."""
    line = "=" * 60
    print()
    print(line)
    print("  Beyond Spatial Resolution")
    print("  Agricultural Plastic Structures Mapping")
    print(line)
    print()


def _print_all_done(results_path):
    """Print the final completion message."""
    print()
    print("=" * 60)
    print(f"  [Success]  All done!  ->  {results_path}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    cfg  = resolve_config(args)

    # -- Welcome ---------------------------------------------------------
    _print_banner()
    print("Initializing...\n")

    model_names = cfg["enabled_models"] if cfg["enabled_models"] else ALL_MODELS
    exp_list    = cfg["selected_experiments"]
    exp_names   = [e["name"] for e in exp_list]
    any_img     = any(e["apply_model_on_image"] for e in exp_list)

    print(f"  Random state  : {cfg['random_state']}")
    print(f"  Models        : {', '.join(model_names)}  ({len(model_names)} total)")
    print(f"  Experiments   : {', '.join(exp_names)}")
    print(f"  Image classif.: {'enabled' if any_img else 'disabled'}")
    print(f"  Figures       : {'skipped' if cfg['skip_plots'] else 'enabled'}")
    print()

    # Build step plan
    plan_steps = [
        "Load datasets",
        f"Run experiments  ({len(exp_list)} exp × {len(model_names)} model(s)"
        + ("  [+image]" if any_img else "") + ")",
    ]
    if not cfg["skip_plots"]:
        plan_steps.append("Generate figures")
    total_steps = len(plan_steps)

    print("Planned steps:")
    for i, s in enumerate(plan_steps, 1):
        print(f"  [{i}] {s}")
    print()
    print("-" * 60)

    # -- Font (silent) --------------------------------------------
    try:
        rc("font", family="Times New Roman")
    except Exception:
        pass

    # -- Path setup ----------------------------------------------
    base, datasets_dir, results_dir = resolve_base_path()

    # Adjust sys.path so that `from src.xxx import ...` works in all layouts
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    from src.config import build_models
    from src.paths import setup_paths
    from src.data import load_datasets
    from src.experiments import run_all_experiments
    from src.plotting import (
        load_plot_data,
        plot_scatter_from_mode,
        plot_oa_kappa_bars,
        plot_satellite_impact,
        plot_band_impact,
        plot_scatter_combined,
    )

    paths = setup_paths(base=base)

    # Override dataset/results paths when running on Code Ocean
    if datasets_dir != base / "data":
        paths["datasets_path"] = datasets_dir
        paths["results_path"]  = results_dir
        paths["figures_path"]  = results_dir / "figures"
        paths["maps_path"]     = results_dir / "maps"
        paths["figures_path"].mkdir(parents=True, exist_ok=True)
        (paths["maps_path"] / "PNG").mkdir(parents=True, exist_ok=True)
        (paths["maps_path"] / "TIFF").mkdir(parents=True, exist_ok=True)

    # -- Step 1: Load datasets ----------------------------------
    print(f"\n[1/{total_steps}] Loading datasets...")
    datasets = load_datasets(paths)
    print("    [OK] Datasets loaded.\n")

    # -- Step 2: Run experiments --------------------------------
    print(f"[2/{total_steps}] Running experiments...")
    models  = build_models(cfg["random_state"], cfg["enabled_models"])
    results = run_all_experiments(cfg["selected_experiments"], models, datasets, paths)
    print(f"\n    [OK] All experiments complete.")
    print(f"      Results -> {paths['results_path'] / 'model_results.xlsx'}")

    # -- Step 3 (optional): Generate figures ---------------------
    if cfg["skip_plots"]:
        _print_all_done(paths["results_path"])
        return

    RESULTS_FILE = paths["results_path"] / "model_results.xlsx"
    if not RESULTS_FILE.exists():
        print(f"\n    [!]  Results file not found at {RESULTS_FILE}. Skipping figures.")
        _print_all_done(paths["results_path"])
        return

    print(f"\n[3/{total_steps}] Generating figures...")
    plot_data = load_plot_data(RESULTS_FILE)

    _plot_calls = [
        ("Figure1",          lambda: plot_scatter_from_mode(
                                 plot_data, paths["figures_path"],
                                 mode="both", output_name="Figure1.png", bands=True,
                             )),
        ("OA/Kappa bars",    lambda: plot_oa_kappa_bars(RESULTS_FILE,   paths["figures_path"])),
        ("Satellite impact", lambda: plot_satellite_impact(RESULTS_FILE, paths["figures_path"])),
        ("Band impact",      lambda: plot_band_impact(RESULTS_FILE,      paths["figures_path"])),
        ("Scatter combined", lambda: plot_scatter_combined(plot_data,    paths["figures_path"])),
    ]

    _saved, _skipped = [], []
    for _name, _fn in _plot_calls:
        try:
            _fn()
            _saved.append(_name)
        except Exception as _e:
            _skipped.append(_name)
            print(f"    [!]  {_name} skipped - {_e}")

    if _saved:
        print(f"    [OK] Figures saved: {', '.join(_saved)}")
    if _skipped:
        print(f"    [!]  Figures skipped (need more experiments): {', '.join(_skipped)}")

    _print_all_done(paths["results_path"])


if __name__ == "__main__":
    import datetime
    import traceback as _tb

    # Resolve results directory early (same logic as resolve_base_path)
    _base_dir = Path(__file__).resolve().parent
    _results_dir = (
        Path("/results")
        if Path("/data").exists() and Path("/results").exists()
        else _base_dir / "results"
    )
    _results_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _base_dir / "run.log"

    _log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
    _header = (
        f"Beyond Spatial Resolution - Run log\n"
        f"Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'-' * 60}\n"
    )
    _log_file.write(_header)

    sys.stdout = _LogTee(sys.__stdout__, _log_file)
    sys.stderr = _LogTee(sys.__stderr__, _log_file)

    _exit_code = 0
    try:
        main()
    except SystemExit as _e:
        _exit_code = _e.code if isinstance(_e.code, int) else 1
    except Exception:
        _tb.print_exc()          # written to _LogTee → goes to log + console
        _exit_code = 1
    finally:
        _footer = f"{'-' * 60}\nEnded : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        _log_file.write(_footer)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_file.close()
        status = "Success" if _exit_code == 0 else "Failed"
        # Print using standard ascii arrows and checkmarks to avoid Windows charmap errors
        print(f"  [{status}]  |  Log saved -> {_log_path}")

    sys.exit(_exit_code)
