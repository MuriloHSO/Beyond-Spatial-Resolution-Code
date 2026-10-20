# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Agricultural Plastic Remote Mapping
This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Agricultural Plastic Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe), presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/) and now under review in IEEE GRSL. The paper investigates the trade-off between the higher spatial and temporal resolutions of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure
`code/`
- `config.toml` — Sets which code to run: full or quick. It also allows for the selection of models and band configurations.
- `run.py` — Main entry point. Reads `config.toml`; command-line arguments override it (see `--help`). Checks the installed package versions against `requirements.txt`.
- `run.sh` — Bash entry point required by Code Ocean. Calls `run.py` and forwards any CLI arguments.
- `requirements.txt` - lists the required Python packages, with pinned versions (Python 3.12 or newer).
- `src/` - Python package containing all library code:
    - `config.py` — builds the classifiers dictionary
    - `paths.py` — filesystem paths and output directory creation
    - `data.py` — CSV dataset loading
    - `download.py` — download the image dataset from Hugging Face (pinned dataset revision)
    - `preprocessing.py` — 3-time-series composites (30-day means) of PS and S2 from raw scenes (complete run)
    - `download_s2_gee.py` — S2 scenes with their Cloud Score+ cloud score from Google Earth Engine (complete run)
    - `generate_data_points.py` — training/validation points from the reference polygons (complete run)
    - `training.py` — `train_and_evaluate_model` function
    - `tuning.py` — hyperparameter search spaces and Successive Halving tuning
    - `experiments.py` — per-experiment runners and dispatcher
    - `plotting.py` — all figure-generation functions

`data/` - input data directory (mirrors Code Ocean's `/data`):
- CSV training and validation datasets (included in the repository). The `Class` column holds the class code of each point (0 = plastic-mulched farmland, PMF; 1 = plastic greenhouses, PGH; 2 = others); the task is a single three-class problem (plastic-mulched farmland, plastic greenhouses and others) solved by one multiclass model per experiment.
- `Imagery/` — GeoTIFF imagery from [Hugging Face](https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code) (not included; about 8.4 GB). `run.py` downloads it when full-image classification is enabled; to download it alone, run `python -m src.download` from the `code` folder.

`results/` - output directory for all generated files (mirrors Code Ocean's `/results`):
- Classification maps (PNG and TIFF), metric tables, and figures
- `sample_counts.csv` — number of training and validation points per sensor and class
- `model_results_tuned.csv` — OA, Cohen's kappa and macro F1 on the validation set with their 95 % confidence intervals (`CI Low`, `CI High`), training time (TT), image inference speed (IS) and selected hyperparameters of every model
- `class_metrics_tuned.csv` — per-class precision, recall, F1 (with confidence intervals) and support on the validation set
- `sensor_differences_tuned.csv` — OA difference S2 − PS for the same model and band set, with a paired confidence interval
- `validation_predictions_tuned.csv.gz` — class predicted by every model for every validation point (`Point` = row of the validation CSV), from which all metrics and intervals can be recomputed
- `Models/` — every fitted model: `<imagery>_<model>.joblib` (scaler + model, lzma-compressed; load only trusted files, with the pinned scikit-learn) and `<imagery>_<model>.json` (all hyperparameters, features, scaler statistics, seed and versions, enough to refit the model exactly). Only the JSON files are tracked in git
- `confusion_matrices_tuned.csv` — confusion matrix of every model on the validation set (long format: true class, predicted class, count); `figures/Fig_B.7_confusion_matrices.png` shows those of the most accurate classifier of each experiment

`scratch/` - temporary and internal files, not part of the results (git-ignored), e.g. the breakdown of the training time (`time_breakdown_tuned.csv`).

`environment/Dockerfile` and `.codeocean/environment.json` - computational environment of the Code Ocean capsule (managed by Code Ocean).

## Quick Start
1. Clone this repository: 
```bash
git clone https://github.com/MuriloHSO/Beyond-Spatial-Resolution-Code.git
```
2. Install the pinned packages (Python 3.12 or newer):
```bash
pip install -r code/requirements.txt
```
3. Open [`code/config.toml`](code/config.toml) and choose your models, experiments and settings.
4. Run from the `code` folder:
```bash
cd code
python run.py
```
5. The repository is structured for a quick run. To run the complete experiment, including testing all models, downloading imagery from Hugging Face, and inferencing images to determine inference speed (IS), alter the configuration in `config.toml`.

## Configuration (`config.toml`)
All user-facing settings live in [`code/config.toml`](code/config.toml):

| Key | Description |
|-----|-------------|
| `random_state` | Single integer seed for the classifiers, cross-validation splits and hyperparameter search (default 2025) |
| `models` | List of models to run (empty list = all seven) |
| `skip_plots` | Set to `true` to skip figure generation |
| `apply_model_on_image` | `true` to classify the full satellite image in every experiment; `false` to skip (faster) |
| `experiments` | List of experiment names to run |
| `[tuning] enabled` | `true` to tune every model's hyperparameters before evaluation; `false` to use scikit-learn defaults (also `--no-tuning`) |
| `[tuning] n_candidates` | Random hyperparameter configurations sampled per model |
| `[tuning] factor` | Successive Halving elimination rate between rounds |
| `[tuning] cv_folds` | Inner stratified CV folds used to score each configuration |
| `[uncertainty] n_resamples` | Bootstrap resamples for the confidence intervals of the validation metrics |
| `[uncertainty] confidence` | Coverage of the percentile confidence intervals |
| `[data_generation] preprocess` | `true` to build the composites from the raw scenes (also `--preprocess`; default `false`) |
| `[data_generation] generate_points` | `true` to regenerate the four CSVs from the polygons and composites (also `--generate-points`; default `false`) |
| `[data_generation] first_year`, `last_year` | Reference dates are Jun-18 and Dec-15 of every year in this range |
| `[data_generation] windows` | Start and end (exclusive) of the t0, t1, t2 windows, in days from the reference date |
| `[data_generation] training_points`, `validation_points` | Points per class in each set |
| `[data_generation.sentinel2]` | S2 scene cloud filter (per window, from 20 % rising by 10 % until 3 days cover the ROI) and Cloud Score+ clear threshold (`cs_cdf_threshold`) |

Valid model names: `CART`, `KNN`, `MLP`, `RF`, `SGDC`, `LinearSVC`, `SVC_rbf`  
Valid experiment names: `S2_4b`, `S2_Allb`, `PS_4b`, `PS_Allb`

## Data generation (complete run)
The CSVs in `data/` can be rebuilt from raw scenes and reference polygons. These steps are off by default and need the packages in `code/requirements-data.txt`:

1. Place the reference polygons (`polygons/*.gpkg` of the Hugging Face dataset) in `data/polygons/`.
2. Place the 8-band PlanetScope surface reflectance scenes (`*_AnalyticMS_SR_8b*.tif`, ordered from Planet as individual PSScene items with the `analytic_8b_sr_udm2` bundle, without the composite tool) anywhere under `data/raw/PlanetScope/`, each next to its UDM2 mask (`*_udm2*.tif`). The file name must start with the acquisition date (`YYYYMMDD`), as in Planet's naming. Pixels that UDM2 does not mark as clear are left out of the means.
3. Run `python run.py --preprocess --generate-points` from `code/` (or set the switches in `config.toml`). A reference date is used when every window of every ROI has PlanetScope scenes. Missing Sentinel-2 scenes are downloaded from Google Earth Engine after confirmation (`python -m src.download_s2_gee` does it alone); this needs an Earth Engine account, with the Cloud project id in the `EE_PROJECT` environment variable or typed when asked.

Composites are written to `data/composites/` and the four CSVs in `data/` are replaced; they gain the columns `Reference_Date` and `Polygon_ID` (the number, GeoPackage `fid`, of the labelled polygon a PMF or PGH point was drawn from, in the layer of its class; empty for "others"). Each CSV row is a random point inside one 10 m S2 pixel (one point per pixel and reference date) whose values come from the S2 pixel and from the 3 m PS pixel at the point.

## Hyperparameter tuning
Every model is tuned with Successive Halving ([`HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide)) before it is evaluated. For each model, `n_candidates` configurations are sampled from the search space in `src/tuning.py` and scored by stratified, spatially grouped cross-validation on a small subsample of the training set. Only the best `1/factor` of them advance to the next round, which uses `factor` times more samples, until the remaining configurations are scored on the full training set. The highest-scoring configuration (by overall accuracy) is then trained and evaluated as before.

Training points are only a few metres apart, so neighbouring points often fall in the same pixel. With random folds, these near-duplicates end up on both sides of a split and the search favours configurations that memorise them. The inner folds are therefore grouped by labelled polygon and date (`Polygon_ID`, `Reference_Date`): all points of a polygon on one date stay in the same fold; each "others" point is its own group. The same field on another date is an independent group.

MLP and LinearSVC use a larger maximum number of iterations than the scikit-learn defaults (`max_iter` 1000 and 10,000, set in `src/config.py`): with the defaults (200 and 1000), MLP stopped before converging in some image sets and LinearSVC results varied with the random seed.

Tuning uses only the training set; the validation set is never seen during the search. Standardisation is fit inside each inner fold to avoid leakage. Tuned results are saved to `model_results_tuned.csv`, so `model_results.csv` keeps the default-hyperparameter results. In the tuned results, `Training Time (s)` is the time to obtain a trained model: the hyperparameter search plus the final fit (mean of 3 fits), and `Best Params` lists the selected hyperparameters. All times are elapsed times with every CPU thread available: the search runs its candidates in parallel, and the final model uses all threads where the estimator supports it (`n_jobs`). The breakdown of the training time (search, final fit and inner-CV OA) is written to `scratch/time_breakdown_tuned.csv`, outside `results/`.

## Confidence intervals
Error bars and intervals describe the uncertainty of the validation metrics. Points of the same labelled polygon on one date are a few metres apart and not independent, so they are resampled together: a class-stratified bootstrap (`src/uncertainty.py`) draws, for each class, its polygons (one per date) and single "others" points with replacement, 2,000 times, and reports 95 % percentile intervals. The same resamples serve every model and experiment, so the S2 − PS differences get paired intervals.

## Declaration of AI-assisted coding
We refactored the original notebook (.ipynb) code initially used for the paper into a version that meets Code Ocean's reproducibility requirements, using the free versions of Gemini 2.0 and GitHub Copilot (from June/July 2026).


