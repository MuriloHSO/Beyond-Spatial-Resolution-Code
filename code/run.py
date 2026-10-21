"""
run.py
│
├── parse config.toml
├── setup_paths()
├── ensure_composites()
├── load_datasets()
├── run_all_experiments()
├── generate_figures()
├── write_metadata(...)
└── finish()
------
Configuration is read from config.toml in the same directory as this file.
Edit that file to change models, experiments, and other settings, then run:

    python run.py

Command-line arguments can still override any config.toml value:

    python run.py --help

Options
-------
    --config PATH               Path to a TOML config file
                                (default: config.toml next to this script)
    --random-state INT          Random seed for the classifiers, CV splits and
                                hyperparameter search
    --models MODEL [MODEL ...]  Models to run; omit to use config.toml value.
                                Valid: CART KNN MLP RF SGDC LinearSVC SVC_rbf
    --experiments EXP [EXP ...] Experiments to run; omit to use config.toml.
                                Valid: S2_4b S2_Allb PS_4b PS_Allb
    --apply-on-image            Force full-image classification for every
                                experiment (overrides config.toml per-entry flag)
    --no-image                  Disable full-image classification for every
                                experiment (overrides config.toml per-entry flag)
    --skip-plots                Do not generate figures after the experiments.
    --no-tuning                 Use default hyperparameters (skip the
                                Successive Halving search in config.toml)
    --preprocess                Build the PS and S2 composites from the raw
                                scenes in data/raw/ (complete run)
    --generate-points           Regenerate the training/validation CSVs from
                                the polygons and composites (complete run)
"""

import os
import subprocess
import sys
from pathlib import Path


def check_requirements(filename="requirements.txt"):
    """Compare installed package versions with the pins in *filename*; install only if they differ."""
    from importlib.metadata import PackageNotFoundError, version

    req_file = Path(__file__).parent / filename
    if not req_file.exists():
        return

    mismatches = []
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, pinned = line.partition("==")
            try:
                installed = version(name)
            except PackageNotFoundError:
                installed = None
            if pinned and installed != pinned:
                mismatches.append(f"{name} (installed {installed}, required {pinned})")

    if mismatches:
        print(f"Packages differ from {filename}:\n  " + "\n  ".join(mismatches))
        print("Installing the required versions...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("Dependencies installed successfully.\n")

check_requirements()

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for Code Ocean
from matplotlib import rc

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # also silence parallel (joblib) worker processes

# ---------------------------------------------------------------------------
# TOML loading (tomllib is part of the standard library; Python 3.12+ required)
# ---------------------------------------------------------------------------

import tomllib


def load_toml_config(path: Path) -> dict:
    """Load a TOML config file. Returns an empty dict if the file is missing."""
    if not path.exists():
        print(f"WARNING: Config file not found at '{path}'. Using defaults.")
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

ALL_MODELS = ["CART", "KNN", "MLP", "RF", "SGDC", "LinearSVC", "SVC_rbf"]
ALL_EXPERIMENTS = ["S2_4b", "S2_Allb", "PS_4b", "PS_Allb"]

# Fallback defaults (used when config.toml is missing and no CLI flag is given)
_FALLBACK_RANDOM_STATE = 2025  # same as src.config.DEFAULT_RANDOM_STATE
_FALLBACK_MODELS: list = []          # empty → all models
_FALLBACK_TUNING = {
    "enabled": True,
    "n_candidates": 81,
    "factor": 3,
    "cv_folds": 5,
}
_FALLBACK_UNCERTAINTY = {
    "n_resamples": 2000,
    "confidence": 0.95,
}
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
        help="Random seed for the classifiers, CV splits and hyperparameter search.",
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
    parser.add_argument(
        "--no-tuning", action="store_true", default=False,
        help="Use default hyperparameters instead of the Successive Halving search.",
    )
    parser.add_argument(
        "--preprocess", action="store_true", default=False,
        help="Build the PS and S2 composites from the raw scenes in data/raw/ (complete run).",
    )
    parser.add_argument(
        "--generate-points", action="store_true", default=False,
        help="Regenerate data/{PS,S2}_{Training,Validation}.csv from the polygons and composites (complete run).",
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

    # --- hyperparameter tuning ---
    tuning = {**_FALLBACK_TUNING, **toml.get("tuning", {})}
    uncertainty = {**_FALLBACK_UNCERTAINTY, **toml.get("uncertainty", {})}
    if args.no_tuning:
        tuning["enabled"] = False

    # --- data generation (complete run only) ---
    data_generation = dict(toml.get("data_generation", {}))
    data_generation["preprocess"] = args.preprocess or data_generation.get("preprocess", False)
    data_generation["generate_points"] = args.generate_points or data_generation.get("generate_points", False)

    return {
        "random_state": random_state,
        "enabled_models": enabled_models,
        "selected_experiments": selected_experiments,
        "skip_plots": skip_plots,
        "tuning": tuning,
        "uncertainty": uncertainty,
        "data_generation": data_generation,
    }


# ---------------------------------------------------------------------------
# Path resolution (Code Ocean vs. local)
# ---------------------------------------------------------------------------


def resolve_base_path():
    """
    Return the repository root, accounting for Code Ocean's layout.

    Code Ocean mounts:
      /code    — the capsule code directory (repo root)
      /data    — input datasets, composites and raw data
      /results — output directory

    Local layout mirrors Code Ocean conventions:
      <repo>/data/    — CSVs and the composites/, raw/ and polygons/ subfolders
      <repo>/results/ — output directory
    """
    code_ocean_data    = Path("../data")
    code_ocean_results = Path("/results")

    if code_ocean_data.exists() and code_ocean_results.exists():
        # Running inside Code Ocean
        return Path("/code"), code_ocean_data, code_ocean_results

    # Local execution: mirror Code Ocean conventions relative to this file's parent.
    base = Path(__file__).resolve().parent
    return base.parent, base.parent / "data", base.parent / "results"


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
    tuning = cfg["tuning"]
    if tuning["enabled"]:
        print(
            f"  Tuning        : Successive Halving  ({tuning['n_candidates']} candidates, "
            f"factor {tuning['factor']}, {tuning['cv_folds']}-fold CV "
            f"grouped by labelled polygon and date)"
        )
    else:
        print("  Tuning        : disabled (default hyperparameters)")
    uncertainty = cfg["uncertainty"]
    print(
        f"  Intervals     : {uncertainty['confidence']:.0%} class-stratified bootstrap of the validation "
        f"points by polygon ({uncertainty['n_resamples']} resamples)"
    )
    print(f"  Figures       : {'skipped' if cfg['skip_plots'] else 'enabled'}")
    data_generation = cfg["data_generation"]
    data_steps = [
        step for step, key in (("Build composites from raw scenes", "preprocess"),
                               ("Generate data points", "generate_points"))
        if data_generation[key]
    ]
    print(f"  Data generation: {', '.join(data_steps) if data_steps else 'disabled (data/*.csv used as is)'}")
    print()

    # Build step plan
    plan_steps = [
        *data_steps,
        "Load datasets",
        f"Run experiments  ({len(exp_list)} exp × {len(model_names)} model(s)"
        + ("  [+tuning]" if tuning["enabled"] else "")
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

    from src.config import CLASS_NAMES, build_models
    from src.paths import setup_paths
    from src.download import ensure_composites
    from src.data import load_datasets, sample_counts
    from src.experiments import results_filename, run_all_experiments
    from src.plotting import (
        load_plot_data,
        plot_scatter_from_mode,
        plot_oa_kappa_bars,
        plot_satellite_impact,
        plot_band_impact,
        plot_scatter_combined,
        plot_confusion_matrices,
    )

    paths = setup_paths(base=base)
    
    # Override dataset/results paths when running on Code Ocean
    if datasets_dir != base / "data":
        paths["datasets_path"] = datasets_dir
        data = datasets_dir.resolve()
        paths.update({
            "data_path": data,
            "raw_path": data / "raw",
            "PS_raw": data / "raw" / "PlanetScope",
            "S2_raw": data / "raw" / "Sentinel-2",
            "polygons_path": data / "polygons",
            "composites_path": data / "composites",
            "PS_composites": data / "composites" / "PlanetScope",
            "S2_composites": data / "composites" / "Sentinel-2",
        })
        paths["results_path"]  = results_dir
        paths["figures_path"]  = results_dir / "figures"
        paths["maps_path"]     = results_dir / "maps"
        paths["figures_path"].mkdir(parents=True, exist_ok=True)
        (paths["maps_path"] / "PNG").mkdir(parents=True, exist_ok=True)
        (paths["maps_path"] / "TIFF").mkdir(parents=True, exist_ok=True)
        if Path("/scratch").exists():  # Code Ocean's temporary folder
            paths["scratch_path"] = Path("/scratch")

    # Ensure the composites are available only if any experiment requires image classification
    if any_img:
        ensure_composites(paths)
    else:
        print("[OK] Full-image classification disabled. Skipping composites download.")

    step = 0

    # -- Optional steps: data generation (complete run) -----------
    if data_steps:
        check_requirements("requirements-data.txt")
        from src.preprocessing import settings
        data_settings = settings(data_generation)
    if data_generation["preprocess"]:
        from src.preprocessing import build_all
        step += 1
        print(f"\n[{step}/{total_steps}] Building composites...")
        report = build_all(paths, data_settings)
        report.to_csv(paths["scratch_path"] / "preprocessing_report.csv", index=False)
        print("    [OK] Composites built (report -> preprocessing_report.csv in scratch).")
    if data_generation["generate_points"]:
        from src.generate_data_points import generate
        step += 1
        print(f"\n[{step}/{total_steps}] Generating data points...")
        summary = generate(paths, data_settings, cfg["random_state"])
        summary.to_csv(paths["scratch_path"] / "data_points_summary.csv", index=False)
        print("    [OK] Data points written (summary -> data_points_summary.csv in scratch).")

    # -- Step: Load datasets -------------------------------------
    step += 1
    print(f"\n[{step}/{total_steps}] Loading datasets...")
    datasets = load_datasets(paths)
    sample_counts(datasets).to_csv(paths["results_path"] / "sample_counts.csv", index=False)
    print("    [OK] Datasets loaded (sample counts -> sample_counts.csv).\n")

    # -- Step: Run experiments ----------------------------------
    step += 1
    print(f"[{step}/{total_steps}] Running experiments...")
    models  = build_models(cfg["random_state"], cfg["enabled_models"])
    results = run_all_experiments(
    cfg["selected_experiments"],
    models,
    datasets,
    paths,
    tuning=tuning,
    random_state=cfg["random_state"],
    uncertainty=cfg["uncertainty"],
    )
    print(f"\n    [OK] All experiments complete.")
    RESULTS_FILE = paths["results_path"] / results_filename(tuning)
    CONFUSION_FILE = paths["results_path"] / results_filename(tuning).replace("model_results", "confusion_matrices")
    print(f"      Results -> {RESULTS_FILE}")

    # -- Step 3 (optional): Generate figures ---------------------
    if cfg["skip_plots"]:
        _print_all_done(paths["results_path"])
        return

    if not RESULTS_FILE.exists():
        print(f"\n    [!]  Results file not found at {RESULTS_FILE}. Skipping figures.")
        _print_all_done(paths["results_path"])
        return

    step += 1
    print(f"\n[{step}/{total_steps}] Generating figures...")
    plot_data = load_plot_data(RESULTS_FILE)

    _plot_calls = [
        ("Figure1",          lambda: plot_scatter_from_mode(
                                 plot_data, paths["figures_path"],
                                 mode="both", output_name="Figure_1.png", bands=True,
                             )),
        ("Fig_B.1_Scatter combined", lambda: plot_scatter_combined(plot_data,    paths["figures_path"])),
        ("Fig_B.2_Satellite impact", lambda: plot_satellite_impact(RESULTS_FILE, paths["figures_path"])),
        ("Fig_B.3_Band impact",      lambda: plot_band_impact(RESULTS_FILE,      paths["figures_path"])),
        ("Fig_B.4_OA/Kappa bars",    lambda: plot_oa_kappa_bars(RESULTS_FILE,   paths["figures_path"])),
        ("Fig_B.7_Confusion matrices", lambda: plot_confusion_matrices(
                                 RESULTS_FILE, CONFUSION_FILE, paths["figures_path"], CLASS_NAMES,
                             )),
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

    # Resolve the repository data/results layout early so logs and outputs
    # remain in sibling directories outside the code folder.
    _base_dir, _datasets_dir, _results_dir = resolve_base_path()
    _results_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _results_dir / "run.log"

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
