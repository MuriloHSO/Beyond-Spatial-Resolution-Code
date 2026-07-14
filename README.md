# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping

This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe) presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/). The paper investigates the trade-off between the higher spatial resolution of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure

`run.py` — Main entry point. Edit the `DEFAULT_*` constants at the top, or pass command-line arguments (see `--help`).

`run.sh` — Bash entry point required by Code Ocean. Calls `run.py` and forwards any CLI arguments.

`src/` - Python package containing all library code:
- `config.py` — builds the classifiers dictionary
- `paths.py` — filesystem paths and output directory creation
- `data.py` — CSV dataset loading
- `training.py` — `train_and_evaluate_model` function
- `experiments.py` — per-experiment runners and dispatcher
- `plotting.py` — all figure-generation functions

`data/` - input data directory (mirrors Code Ocean's `/data`):
- CSV training and validation datasets (committed to the repository)
- `Imagery/` — must contain the GeoTIFF imagery downloaded from [Hugging Face](https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code) (not committed)

`results/` - output directory for all generated files (mirrors Code Ocean's `/results`):
- Classification maps (PNG and TIFF), metric tables, and figures

`scratch/` - temporary working directory for large intermediate files.

`requirements.txt` - lists the required Python packages to run the code.

## Quick Start
1. Clone this repository: 
```bash
git clone https://github.com/MuriloHSO/Beyond-Spatial-Resolution-Code.git
```
2. Download GeoTIFF imagery from [Hugging Face](https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code) and place in `data/Imagery/`.
```bash
pip install -r requirements.txt
```
3. Open [`config.toml`](config.toml) and choose your models, experiments and settings.
4. Run:
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
| `apply_model_on_image` | `true` to classify the full satellite image in every experiment; `false` to skip (faster) |
| `experiments` | List of experiment names to run |

Valid model names: `CART`, `KNN`, `MLP`, `RF`, `SGD`, `SVM_linear`, `SVM_rbf`  
Valid experiment names: `S2_4b`, `S2_Allb`, `PS_4b`, `PS_Allb`
