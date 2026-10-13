# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Agricultural Plastic Remote Mapping
This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Agricultural Plastic Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe), presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/) and now under review in IEEE GRSL. The paper investigates the trade-off between the higher spatial and temporal resolutions of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure
`code/`
- `config.toml` — Sets which code to run: full or quick. It also allows for the selection of models and band configurations.
- `run.py` — Main entry point. Edit the `DEFAULT_*` constants at the top, or pass command-line arguments (see `--help`).
- `run.sh` — Bash entry point required by Code Ocean. Calls `run.py` and forwards any CLI arguments.
- `requirements.txt` - lists the required Python packages to run the code.
- `src/` - Python package containing all library code:
    - `config.py` — builds the classifiers dictionary
    - `paths.py` — filesystem paths and output directory creation
    - `data.py` — CSV dataset loading
    - `download.py` — download image dataset from Hugging Face
    - `training.py` — `train_and_evaluate_model` function
    - `tuning.py` — hyperparameter search spaces and Successive Halving tuning
    - `experiments.py` — per-experiment runners and dispatcher
    - `plotting.py` — all figure-generation functions

`data/` - input data directory (mirrors Code Ocean's `/data`):
- CSV training and validation datasets (included in the repository)
- `Imagery/` — must contain the GeoTIFF imagery downloaded from [Hugging Face](https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code) (not includeded)

`results/` - output directory for all generated files (mirrors Code Ocean's `/results`):
- Classification maps (PNG and TIFF), metric tables, and figures

`scratch/` - temporary working directory for large intermediate files.

## Quick Start
1. Clone this repository: 
```bash
git clone https://github.com/MuriloHSO/Beyond-Spatial-Resolution-Code.git
```
2. Open [`config.toml`](config.toml) and choose your models, experiments and settings.
3. Run:
```bash
python run.py
```
4. The repository is structured for a quick run. To run the complete experiment, including testing all models, downloading imagery from Hugging Face, and inferencing images to determine inference speed (IS), alter the configuration in `config.toml`.

## Configuration (`config.toml`)
All user-facing settings live in [`config.toml`](config.toml):

| Key | Description |
|-----|-------------|
| `random_state` | Integer seed for all classifiers |
| `models` | List of models to run (empty list = all seven) |
| `skip_plots` | Set to `true` to skip figure generation |
| `apply_model_on_image` | `true` to classify the full satellite image in every experiment; `false` to skip (faster) |
| `experiments` | List of experiment names to run |
| `[tuning] enabled` | `true` to tune every model's hyperparameters before evaluation; `false` to use scikit-learn defaults (also `--no-tuning`) |
| `[tuning] n_candidates` | Random hyperparameter configurations sampled per model |
| `[tuning] factor` | Successive Halving elimination rate between rounds |
| `[tuning] cv_folds` | Inner stratified CV folds used to score each configuration |
| `[tuning] block_size` | Side (m) of the square spatial blocks used to build the inner CV folds |

Valid model names: `CART`, `KNN`, `MLP`, `RF`, `SGDC`, `LinearSVC`, `SVC_rbf`  
Valid experiment names: `S2_4b`, `S2_Allb`, `PS_4b`, `PS_Allb`

## Hyperparameter tuning
Every model is tuned with Successive Halving ([`HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide)) before it is evaluated. For each model, `n_candidates` configurations are sampled from the search space in `src/tuning.py` and scored by stratified, spatially grouped cross-validation on a small subsample of the training set. Only the best `1/factor` of them advance to the next round, which uses `factor` times more samples, until the remaining configurations are scored on the full training set. The highest-scoring configuration (by overall accuracy) is then trained and evaluated as before.

Training points are only a few metres apart, so neighbouring points often fall in the same pixel. With random folds, these near-duplicates end up on both sides of a split and the search favours configurations that memorise them. The inner folds are therefore built from square spatial blocks of `block_size` metres (1 km by default): all points in a block stay in the same fold.

Tuning uses only the training set; the validation set is never seen during the search. Standardisation is fit inside each inner fold to avoid leakage. Tuned results are saved to `model_results_tuned.csv`, so `model_results.csv` keeps the default-hyperparameter results. The tuned results file reports the search time (`Tuning Time (s)`), the best inner-CV overall accuracy (`Tuning CV OA`) and the selected hyperparameters (`Best Params`). The training, validation and classification times refer to the final tuned model, so they remain comparable with untuned runs.

## Declaration of AI-assisted coding
We refactored the original notebook (.ipynb) code initially used for the paper into a version that meets Code Ocean's reproducibility requirements, using the free versions of Gemini 2.0 and GitHub Copilot (from June/July 2026).


