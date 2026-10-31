# PlanetScope composites: not distributed, build them with the code

The PlanetScope (PS) composites used in the paper are **not shared**. The PS scenes were obtained through Planet's Education and Research (E&R) Basic Program, whose terms allow public sharing only of derivative products that contain no source image data and do not retain Planet's raw data values ([E&R Basic Terms of Service, May 2023](https://assets.planet.com/docs/E&R_Basic_ToS_Revised_May_2023.pdf); [Dataset Sharing Guidelines for the E&R Basic Program](https://support.planet.com/hc/en-us/articles/22540409084701-Dataset-Sharing-Guidelines-for-the-Education-and-Research-Basic-Program); [Restrictions of the Basic E&R Program](https://support.planet.com/hc/en-us/articles/360016148637)). The composites are 3 m surface-reflectance images, so neither they nor the raw scenes are redistributed. The PS point tables (`PS_TrainTest.csv`, `PS_Validation.csv`) hold mean PS reflectance values and are not redistributed either: Planet's support confirmed that, under E&R Basic, such tables cannot be published as an open dataset.

What is shared: the Sentinel-2 composites and point tables, the reference polygons and the classification maps ([Hugging Face dataset](https://huggingface.co/datasets/marlonfs/agricultural-plastic-PS-S2), CC BY-NC 4.0), and the list of PS scenes used, `PS_scenes.csv` in this folder (Planet item ids per ROI, reference date and window).

## Build the PS composites with your own Planet account

1. You need a Planet account with access to PlanetScope 8-band surface reflectance (for example, Planet's Education and Research Program or a commercial licence). Your use of the data is governed by your own agreement with Planet.
2. Order the items listed in `PS_scenes.csv` (item type `PSScene`, bundle `analytic_8b_sr_udm2`: 8-band surface reflectance with its usable data mask UDM2), GeoTIFF, with the **Clip** tool only (no Composite tool). Clip to each ROI: the rectangles in `polygons/ROIs.gpkg` of the Hugging Face dataset (training and validation halves together), with a 30 m margin. Instead of the item list, you can search Planet's catalogue in the windows given in the CSV (`window_first_day` to `window_last_day`, both included), SuperDove (PSB.SD) only.
3. Unzip the orders anywhere under `data/raw/PlanetScope/`, keeping Planet's file names (`<YYYYMMDD>_..._AnalyticMS_SR_8b_clip.tif` next to its `..._udm2_clip.tif`). Put the polygons in `data/polygons/`.
4. Install the packages in `code/requirements-data.txt` and run, from the `code` folder:

       python -m src.preprocessing PS

   This writes `PS_<ROI>_<YYYY-MM-DD>.tif` here (about 7.9 GB; about 1 h on a laptop) with the same settings as the paper (`[data_generation]` in `config.toml`): the 30-day windows and the UDM2 `clear` mask. No Earth Engine account is needed for PS. The Sentinel-2 composites are downloaded with `python -m src.download`.
5. Rebuild the PS point tables from the composites and the polygons with `python run.py --generate-points` (see "Data generation" in the main README). With `apply_model_on_image = true` in `config.toml`, `run.py` then also classifies the full PS composites.

Planet may reprocess archive scenes, so rebuilt values, and the results computed from them, can differ slightly from those in the paper.
