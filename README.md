# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping

This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe) presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/). The paper investigates the trade-off between the higher spatial resolution of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure

`run.py` — Main entry point. Edit the `DEFAULT_*` constants at the top, or pass command-line arguments (see `--help`).

`src/` - Python package containing all library code:
- `config.py` — builds the classifiers dictionary
- `paths.py` — filesystem paths and output directory creation
- `data.py` — CSV dataset loading
- `training.py` — `train_and_evaluate_model` function
- `experiments.py` — per-experiment runners and dispatcher
- `plotting.py` — all figure-generation functions

`Datasets/` - contains the data used for data processing and evaluation.

`Imagery/` - must contain the imagery used for classification and test, that can be downloaded at https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code.

`Results/` - contains the results of data processing and evaluation.

`requirements.txt` - lists the required Python packages to run the code.

## Quick Start

```bash
pip install -r requirements.txt
```

1. Open [`config.toml`](config.toml) and choose your models, experiments and settings.
2. Run:

```bash
python run.py
```

## Configuration (`config.toml`)

All user-facing settings live in [`config.toml`](config.toml):

| Key | Description |
|-----|-------------|
| `random_state` | Integer seed for all classifiers |
| `models` | List of models to run (empty list = all seven) |
| `skip_plots` | Set to `true` to skip figure generation |
| `[[experiments]]` | One block per experiment; set `apply_model_on_image` to control full-image classification |

Valid model names: `CART`, `KNN`, `MLP`, `RF`, `SGD`, `SVM_linear`, `SVM_rbf`  
Valid experiment names: `S2_4b`, `S2_Allb`, `PS_4b`, `PS_Allb`

## Command-Line Overrides

CLI arguments override any value in `config.toml`:

```
python run.py --help

Options:
  --config PATH               Path to an alternative TOML config file
  --random-state INT          Random seed for classifiers
  --models MODEL [MODEL ...]  Models to run
  --experiments EXP [EXP ...] Experiments to run
  --apply-on-image            Force full-image classification for all experiments
  --no-image                  Disable full-image classification for all experiments
  --skip-plots                Skip figure generation
```

### Examples

Run all experiments without image classification (faster):
```bash
python run.py --no-image
```

Run only Sentinel-2 experiments with the Random Forest model:
```bash
python run.py --experiments S2_4b S2_Allb --models RF
```

Use a custom config file:
```bash
python run.py --config my_config.toml
```

## Running on Code Ocean

On Code Ocean, the capsule expects:
- **Input data** mounted at `/data/` (place the contents of `Datasets/` here, and optionally `Imagery/`)
- **Output results** written to `/results/`

The entry point (`run.py`) automatically detects the Code Ocean environment and adjusts paths accordingly.
Set the capsule's **run command** to:
```
python run.py
```
or pass any of the options above.
