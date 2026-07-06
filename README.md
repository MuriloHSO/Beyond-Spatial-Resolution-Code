# Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping

This repository contains the reference code for the paper ["Beyond Spatial Resolution: Comparing Sentinel-2 and PlanetScope Imagery for Efficient Remote Mapping"](https://openreview.net/forum?id=KrttzXQWRe) presented at the [4th Machine Learning for Remote Sensing (ML4RS) Workshop](https://ml-for-rs.github.io/iclr2026/) of [ICLR 2026](https://iclr.cc/). The paper investigates the trade-off between the higher spatial resolution of PlanetScope (PS) and the computational demands associated with its larger data volume, comparing it with Sentinel-2 (S2) in mapping Agricultural Plastic Structures (APS).

## Repository Structure
`Code.ipynb` - contains the source code to reproduce the results presented in the paper. It includes data loading, evaluation steps and figures generation.

`Dataset` - contains the data used for data processing and evaluation.

`Imagery` - must contains the imagery used for classification and test, that can be downloaded in https://huggingface.co/datasets/MuriloHSO/Beyond-Spatial-Resolution-Code.

`Results` - contains the results of data processing and evaluation.

`Requirements.txt` - lists the required Python packages to run the code.
