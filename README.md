# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping

This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe) presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/). The paper investigates the trade-off between the higher spatial resolution of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure

`code.ipynb` - Control panel notebook. Edit only **Cell 0** to choose models, the random state, and whether to apply classification on the full imagery. All other cells run unchanged.

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
jupyter notebook code.ipynb
```

Open `code.ipynb`, edit **Cell 0** as needed, then run all cells.
