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

import argparse
import sys
import warnings
from pathlib import Path

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
# Console output helpers
# ---------------------------------------------------------------------------


def _print_banner():
    """Print a welcome banner to stdout."""
    line = "═" * 60
    print()
    print(line)
    print("  Beyond Spatial Resolution")
    print("  Agricultural Plastic Structures Mapping")
    print(line)
    print()


def _print_all_done(results_path):
    """Print the final completion message."""
    print()
    print("═" * 60)
    print(f"  ✓  All done!  →  {results_path}")
    print("═" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    cfg  = resolve_config(args)

    # ── Welcome ─────────────────────────────────────────────────────────
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
    print("─" * 60)

    # ── Font (silent) ────────────────────────────────────────────
    try:
        rc("font", family="Times New Roman")
    except Exception:
        pass

    # ── Path setup ──────────────────────────────────────────────
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
    if datasets_dir != base / "Datasets":
        paths["datasets_path"] = datasets_dir
        paths["results_path"]  = results_dir
        (results_dir / "PNG").mkdir(parents=True, exist_ok=True)
        (results_dir / "TIFF").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load datasets ──────────────────────────────────
    print(f"\n[1/{total_steps}] Loading datasets...")
    datasets = load_datasets(paths)
    print("    ✓ Datasets loaded.\n")

    # ── Step 2: Run experiments ────────────────────────────────
    print(f"[2/{total_steps}] Running experiments...")
    models  = build_models(cfg["random_state"], cfg["enabled_models"])
    results = run_all_experiments(cfg["selected_experiments"], models, datasets, paths)
    print(f"\n    ✓ All experiments complete.")
    print(f"      Results → {paths['results_path'] / 'model_results.xlsx'}")

    # ── Step 3 (optional): Generate figures ─────────────────────
    if cfg["skip_plots"]:
        _print_all_done(paths["results_path"])
        return

    RESULTS_FILE = paths["results_path"] / "model_results.xlsx"
    if not RESULTS_FILE.exists():
        print(f"\n    ⚠  Results file not found at {RESULTS_FILE}. Skipping figures.")
        _print_all_done(paths["results_path"])
        return

    print(f"\n[3/{total_steps}] Generating figures...")
    plot_data = load_plot_data(RESULTS_FILE)
    plot_scatter_from_mode(
        plot_data, paths["results_path"],
        mode="both", output_name="Figure1.png", bands=True,
    )
    plot_oa_kappa_bars(RESULTS_FILE, paths["results_path"])
    plot_satellite_impact(RESULTS_FILE, paths["results_path"])
    plot_band_impact(RESULTS_FILE, paths["results_path"])
    plot_scatter_combined(plot_data, paths["results_path"])
    print("    ✓ Figures saved.")

    _print_all_done(paths["results_path"])


if __name__ == "__main__":
    main()
