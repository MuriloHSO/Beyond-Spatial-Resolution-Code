"""
download_s2_gee.py

Download Sentinel-2 L2A scenes (COPERNICUS/S2_SR_HARMONIZED) from Google Earth
Engine for every ROI, reference date and composite window, with their Cloud
Score+ quality score (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED; Pasquarella et
al., 2023, CVPR Workshops):
https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED

Each scene is saved on the ROI's 10 m grid (EPSG:32723) as
    data/raw/Sentinel-2/<ROI>/<system:index>.tif
with the 10 bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 (uint16,
reflectance x 10,000; 0 outside the scene) and an 11th band "cs_cdf" (Cloud
Score+ cs_cdf x 10,000, uint16; 0 where there is no score). The mask is applied
when compositing (src/preprocessing.py): a pixel is clear when cs_cdf >=
cs_cdf_threshold (config.toml). Existing files are skipped, so an interrupted
download resumes.

Scene selection per window: the maximum CLOUDY_PIXEL_PERCENTAGE starts at
cloud_filter (config.toml) and rises by cloud_filter_step, up to 100 %, until
at least min_full_coverage_days days have scenes that together cover the ROI.
The selected scenes, with the filter used, are listed in
data/raw/Sentinel-2/scenes.csv; the composites use only these scenes.

Requires a Google Earth Engine account. The Cloud project id is read from the
EE_PROJECT environment variable or asked for.

Run from the code folder:  python -m src.download_s2_gee
"""

import io
import os
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import Window

from src.preprocessing import (
    GRID_CRS,
    SENSORS,
    load_rois,
    plan,
    read_settings,
    roi_grid,
    window_ranges,
)

S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
BANDS = SENSORS["S2"]["bands"]
TILE = 512  # pixels per side of each request (11 bands x 512 x 512 x 2 bytes = 5.8 MB)
WORKERS = 8  # tile requests in parallel
FULL_COVERAGE = 0.99  # share of the ROI a day's scenes must cover to count as full coverage
MANIFEST_COLUMNS = ["roi", "reference_date", "window", "scene", "cloudy_pixel_percentage",
                    "cloud_filter", "full_coverage_days"]


def initialize(project: str | None = None) -> None:
    import ee

    project = project or os.environ.get("EE_PROJECT") or input("Google Earth Engine project id: ").strip()
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


# ---------------------------------------------------------------------------
# Scene selection and Cloud Score+
# ---------------------------------------------------------------------------

def s2_collection(region, start: str, end: str, cloud_filter: float):
    """S2 SR scenes with at most *cloud_filter* % cloudy pixels, joined with their Cloud Score+ image."""
    import ee

    sr = (ee.ImageCollection(S2_COLLECTION_ID)
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_filter)))
    scores = (ee.ImageCollection(CLOUD_SCORE_PLUS_ID)
              .filterBounds(region)
              .filterDate(start, end))
    return ee.ImageCollection(ee.Join.saveFirst("cloud_score").apply(
        primary=sr,
        secondary=scores,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    ))


def full_coverage_days(collection, region) -> int:
    """Days whose scenes together cover at least FULL_COVERAGE of *region*."""
    import ee

    area = region.area(1)
    days = collection.aggregate_array("system:index").map(lambda i: ee.String(i).slice(0, 8)).distinct()
    coverage = days.map(lambda day: collection.filter(ee.Filter.stringStartsWith("system:index", day))
                        .geometry().intersection(region, 1).area(1).divide(area))
    return ee.List(coverage).filter(ee.Filter.gte("item", FULL_COVERAGE)).size().getInfo()


def select_scenes(region, start: str, end: str, p: dict):
    """
    Scenes of one window: the cloud filter starts at p["cloud_filter"] and rises by
    p["cloud_filter_step"] (up to 100 %) until p["min_full_coverage_days"] days cover the ROI.

    Returns the collection, the cloud filter used and the number of full-coverage days.
    """
    cloud_filter = p["cloud_filter"]
    while True:
        collection = s2_collection(region, start, end, cloud_filter)
        days = full_coverage_days(collection, region)
        if days >= p["min_full_coverage_days"] or cloud_filter >= 100:
            return collection, cloud_filter, days
        cloud_filter = min(cloud_filter + p["cloud_filter_step"], 100)


def export_image(img):
    """10 bands (0 outside the scene) and Cloud Score+ cs_cdf x 10,000 (0 where there is no score)."""
    import ee

    cs_cdf = ee.Image(img.get("cloud_score")).select("cs_cdf").multiply(10000).round().unmask(0).rename("cs_cdf")
    return img.select(BANDS).unmask(0).toUint16().addBands(cs_cdf.toUint16())


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_scene(image, grid: dict, out_path: Path) -> None:
    """Write *image* on *grid*, requesting TILE x TILE blocks with computePixels (in parallel)."""
    from concurrent.futures import ThreadPoolExecutor

    import ee

    t = grid["transform"]

    def fetch(block):
        col, row, width, height = block
        request = {
            "expression": image,
            "fileFormat": "GEO_TIFF",
            "grid": {
                "dimensions": {"width": width, "height": height},
                "affineTransform": {
                    "scaleX": t.a, "shearX": 0, "translateX": t.c + col * t.a,
                    "shearY": 0, "scaleY": t.e, "translateY": t.f + row * t.e,
                },
                "crsCode": GRID_CRS,
            },
        }
        with rasterio.open(io.BytesIO(ee.data.computePixels(request))) as tile:
            return block, tile.read()

    blocks = [
        (col, row, min(TILE, grid["width"] - col), min(TILE, grid["height"] - row))
        for row in range(0, grid["height"], TILE)
        for col in range(0, grid["width"], TILE)
    ]
    profile = {
        "driver": "GTiff", "dtype": "uint16", "count": len(BANDS) + 1,
        "width": grid["width"], "height": grid["height"],
        "crs": GRID_CRS, "transform": t,
        "compress": "deflate", "predictor": 2, "tiled": True, "interleave": "band",
        "blockxsize": 512, "blockysize": 512,
    }
    tmp_path = out_path.with_suffix(".tmp.tif")
    with rasterio.open(tmp_path, "w", **profile) as dst, ThreadPoolExecutor(WORKERS) as pool:
        for (col, row, width, height), data in pool.map(fetch, blocks):
            dst.write(data, window=Window(col, row, width, height))
        for i, name in enumerate([*BANDS, "cs_cdf"], start=1):
            dst.set_band_description(i, name)
    tmp_path.replace(out_path)


def download_s2(paths: dict, cfg: dict, rois, dates: list, project: str | None = None) -> pd.DataFrame:
    """Download every S2 scene of *dates* (all windows) for every ROI."""
    import ee

    initialize(project)
    p = cfg["sentinel2"]
    manifest_path = paths["S2_raw"] / "scenes.csv"
    key = ["roi", "reference_date", "window"]
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    for roi, geom in rois.geometry.items():
        grid = roi_grid(geom.bounds, SENSORS["S2"]["resolution"])
        region = ee.Geometry.Rectangle(list(grid["bounds"]), proj=GRID_CRS, geodesic=False)
        folder = paths["S2_raw"] / roi
        folder.mkdir(parents=True, exist_ok=True)
        for ref in dates:
            for t, (start, end) in enumerate(window_ranges(ref, cfg["windows"])):
                collection, cloud_filter, days = select_scenes(region, start.isoformat(), end.isoformat(), p)
                ids = collection.aggregate_array("system:index").getInfo()
                clouds = collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
                warning = "" if days >= p["min_full_coverage_days"] else " [WARN] fewer full-coverage days than required"
                print(f"    {roi} {ref} t{t} ({start} to {end}): cloud filter {cloud_filter} %, "
                      f"{days} full-coverage days, {len(ids)} scenes{warning}")
                for scene_id in ids:
                    out_path = folder / f"{scene_id}.tif"
                    if not out_path.exists():
                        image = export_image(collection.filter(ee.Filter.eq("system:index", scene_id)).first())
                        _download_scene(image, grid, out_path)
                rows = pd.DataFrame({"roi": roi, "reference_date": ref.isoformat(), "window": t, "scene": ids,
                                     "cloudy_pixel_percentage": clouds, "cloud_filter": cloud_filter,
                                     "full_coverage_days": days}, columns=MANIFEST_COLUMNS)
                # Replace any earlier selection of this window, then save (an interrupted run keeps its record).
                same = (manifest[key].astype(str) == [roi, ref.isoformat(), str(t)]).all(axis=1)
                manifest = pd.concat([manifest[~same], rows], ignore_index=True)
                manifest.to_csv(manifest_path, index=False)
    return manifest


if __name__ == "__main__":
    from src.paths import setup_paths

    base = Path(__file__).resolve().parents[2]
    paths = setup_paths(base=base)
    cfg = read_settings(base / "code" / "config.toml")
    table = plan(paths, cfg)
    dates = sorted(table.loc[table["complete"], "reference_date"].unique())
    if not dates:
        raise SystemExit(f"No reference date has PlanetScope scenes in {paths['PS_raw']} for every ROI and window.")
    download_s2(paths, cfg, load_rois(paths), dates)
