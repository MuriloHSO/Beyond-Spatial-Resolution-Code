"""
preprocessing.py

Build the 3-time-series composites of PlanetScope (PS) and Sentinel-2 (S2) for
every region of interest (ROI) and reference date. Each composite stacks three
30-day mean composites (t0, t1, t2) whose windows are set in days from the
reference date (config.toml, [data_generation].windows):

    t0 [-30, 0)   bare soil before planting
    t1 [30, 60)   visible plastic mulch
    t2 [100, 130) full crop canopy

Reference dates are Jun-18 and Dec-15 of every year, the same for both sensors.

Inputs (scenes kept locally, not in the repository):
    data/raw/PlanetScope/**/<YYYYMMDD>_*_AnalyticMS_SR_8b*.tif
        8-band PlanetScope SuperDove surface reflectance, ordered from Planet.
    data/raw/Sentinel-2/<ROI>/<scene>.tif
        10 S2 L2A bands + s2cloudless cloud/shadow mask on the ROI grid,
        downloaded by src/download_s2_gee.py (offered here if missing).
    data/polygons/ROIs.gpkg
        ROI polygons (EPSG:32723) from the Hugging Face dataset.

Output:
    data/composites/{PlanetScope,Sentinel-2}/<PS|S2>_<ROI>_<YYYY-MM-DD>.tif
        uint16 reflectance x 10,000 (nodata 0), bands named <band>_<t>,
        on a fixed grid per ROI (PS 3 m, S2 10 m, EPSG:32723).

Within a window, scenes acquired on the same day are averaged first (overlapping
strips or tiles), then the daily images are averaged, so every day has the same
weight. S2 pixels flagged by the cloud/shadow mask are excluded; PS has no cloud
mask (only clear scenes are ordered).

Run from the code folder:  python -m src.preprocessing
"""

import datetime as dt
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

GRID_CRS = "EPSG:32723"  # WGS 84 / UTM zone 23S, the study area's zone

SENSORS = {
    "PS": {
        "folder": "PlanetScope",
        # PSB.SD: coastal blue, blue, green I, green, yellow, red, red edge, NIR
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        "resolution": 3,
    },
    "S2": {
        "folder": "Sentinel-2",
        "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        "resolution": 10,
    },
}

PS_PATTERN = "*_AnalyticMS_SR_8b*.tif"

DEFAULTS = {
    "first_year": 2020,
    "last_year": 2022,
    "windows": [[-30, 0], [30, 60], [100, 130]],
    "training_points": 12000,
    "validation_points": 4000,
    "sentinel2": {
        "cloud_filter": 50,
        "cloud_probability_threshold": 60,
        "nir_dark_threshold": 0.15,
        "cloud_projection_distance": 2,
        "buffer": 100,
    },
}


# ---------------------------------------------------------------------------
# Settings, dates and grids
# ---------------------------------------------------------------------------

def settings(data_generation: dict | None = None) -> dict:
    """Merge the [data_generation] table of config.toml with the defaults."""
    data_generation = data_generation or {}
    merged = {**DEFAULTS, **data_generation}
    merged["sentinel2"] = {**DEFAULTS["sentinel2"], **data_generation.get("sentinel2", {})}
    return merged


def read_settings(config_path: Path) -> dict:
    with open(config_path, "rb") as fh:
        return settings(tomllib.load(fh).get("data_generation"))


def reference_dates(first_year: int, last_year: int) -> list:
    """Jun-18 and Dec-15 of every year in [first_year, last_year]."""
    return [
        dt.date(year, month, day)
        for year in range(first_year, last_year + 1)
        for month, day in ((6, 18), (12, 15))
    ]


def window_ranges(reference_date: dt.date, windows: list) -> list:
    """(start, end) dates of each window; end is exclusive."""
    return [
        (reference_date + dt.timedelta(days=start), reference_date + dt.timedelta(days=end))
        for start, end in windows
    ]


def load_rois(paths: dict):
    """ROI polygons dissolved per ROI (the training/validation halves joined)."""
    import geopandas as gpd

    rois = gpd.read_file(paths["polygons_path"] / "ROIs.gpkg").to_crs(GRID_CRS)
    return rois.dissolve(by="roi").sort_index()


def roi_grid(bounds, resolution: float) -> dict:
    """Grid covering *bounds*, snapped outward to multiples of *resolution*."""
    left, bottom, right, top = bounds
    left = np.floor(left / resolution) * resolution
    bottom = np.floor(bottom / resolution) * resolution
    right = np.ceil(right / resolution) * resolution
    top = np.ceil(top / resolution) * resolution
    return {
        "transform": from_origin(left, top, resolution, resolution),
        "width": int(round((right - left) / resolution)),
        "height": int(round((top - bottom) / resolution)),
        "crs": GRID_CRS,
        "bounds": (left, bottom, right, top),
    }


def composite_path(paths: dict, sensor: str, roi: str, reference_date: dt.date) -> Path:
    folder = paths[f"{sensor}_composites"]
    return folder / f"{sensor}_{roi}_{reference_date.isoformat()}.tif"


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _scene_date(path: Path) -> dt.date:
    match = re.match(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"No acquisition date (YYYYMMDD) at the start of {path.name}")
    return dt.datetime.strptime(match.group(1), "%Y%m%d").date()


def _bounds_in_grid_crs(src) -> tuple:
    from rasterio.warp import transform_bounds

    return transform_bounds(src.crs, GRID_CRS, *src.bounds)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def find_ps_scenes(paths: dict) -> pd.DataFrame:
    """Every 8-band PS scene under data/raw/PlanetScope, with date and bounds."""
    rows = []
    for path in sorted(paths["PS_raw"].rglob(PS_PATTERN)):
        with rasterio.open(path) as src:
            if src.count != len(SENSORS["PS"]["bands"]):
                raise ValueError(f"{path.name} has {src.count} bands; 8-band SuperDove scenes are required")
            rows.append({"path": path, "date": _scene_date(path), "bounds": _bounds_in_grid_crs(src)})
    return pd.DataFrame(rows, columns=["path", "date", "bounds"])


def find_s2_scenes(paths: dict, roi: str) -> pd.DataFrame:
    """S2 scenes downloaded for *roi* (already on the ROI grid)."""
    folder = paths["S2_raw"] / roi
    rows = [
        {"path": path, "date": _scene_date(path)}
        for path in sorted(folder.glob("*.tif"))
    ] if folder.exists() else []
    return pd.DataFrame(rows, columns=["path", "date"])


def scenes_in_window(scenes: pd.DataFrame, start: dt.date, end: dt.date, roi_bounds=None) -> pd.DataFrame:
    selected = scenes[(scenes["date"] >= start) & (scenes["date"] < end)]
    if roi_bounds is not None and "bounds" in selected:
        selected = selected[[_intersects(b, roi_bounds) for b in selected["bounds"]]]
    return selected


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def _same_crs(crs) -> bool:
    from pyproj import CRS

    return CRS.from_user_input(crs.to_wkt()).to_epsg(min_confidence=70) == 32723


def _read_on_grid(src, band: int, grid: dict) -> np.ndarray:
    """Band *band* of *src* on *grid* (0 outside the scene)."""
    t = src.transform
    g = grid["transform"]
    col_off = (g.c - t.c) / t.a
    row_off = (g.f - t.f) / t.e
    aligned = (
        _same_crs(src.crs)
        and np.isclose(t.a, g.a) and np.isclose(t.e, g.e) and t.b == 0 and t.d == 0
        and np.isclose(col_off, round(col_off)) and np.isclose(row_off, round(row_off))
    )
    if aligned:
        window = Window(round(col_off), round(row_off), grid["width"], grid["height"])
        return src.read(band, window=window, boundless=True, fill_value=0)

    # Scene grid not aligned with the ROI grid: nearest neighbour keeps the original values.
    out = np.zeros((grid["height"], grid["width"]), dtype=src.dtypes[band - 1])
    reproject(
        rasterio.band(src, band), out,
        dst_transform=grid["transform"], dst_crs=grid["crs"],
        resampling=Resampling.nearest, src_nodata=0, dst_nodata=0,
    )
    return out


def _window_mean(scenes: pd.DataFrame, sensor: str, band_index: int, grid: dict) -> np.ndarray:
    """Mean of one band over the days of a window (NaN where no valid observation)."""
    shape = (grid["height"], grid["width"])
    window_sum = np.zeros(shape, dtype=np.float64)
    window_days = np.zeros(shape, dtype=np.uint16)

    for _, day in scenes.groupby("date"):
        day_sum = np.zeros(shape, dtype=np.float64)
        day_n = np.zeros(shape, dtype=np.uint16)
        for path in day["path"]:
            with rasterio.open(path) as src:
                values = _read_on_grid(src, band_index, grid)
                valid = values > 0
                if sensor == "S2":
                    cloudmask = _read_on_grid(src, src.count, grid)  # last band
                    valid &= cloudmask == 0
            day_sum[valid] += values[valid]
            day_n[valid] += 1
        observed = day_n > 0
        window_sum[observed] += day_sum[observed] / day_n[observed]
        window_days[observed] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(window_days > 0, window_sum / window_days, np.nan)


def build_composite(scenes_by_window: list, sensor: str, grid: dict, out_path: Path, tags: dict) -> list:
    """
    Write the 3-time-series composite of one sensor, ROI and reference date.

    Returns the share of pixels with no valid observation in each window.
    """
    bands = SENSORS[sensor]["bands"]
    n_bands = len(bands) * len(scenes_by_window)
    profile = {
        "driver": "GTiff", "dtype": "uint16", "nodata": 0, "count": n_bands,
        "width": grid["width"], "height": grid["height"],
        "crs": grid["crs"], "transform": grid["transform"],
        "compress": "deflate", "predictor": 2, "tiled": True, "interleave": "band",
        "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")
    missing = []
    with rasterio.open(tmp_path, "w", **profile) as dst:
        for t, scenes in enumerate(scenes_by_window):
            no_data = np.zeros((grid["height"], grid["width"]), dtype=bool)
            for b, name in enumerate(bands, start=1):
                mean = _window_mean(scenes, sensor, b, grid)
                no_data |= np.isnan(mean)
                index = t * len(bands) + b
                dst.write(np.where(np.isnan(mean), 0, np.clip(np.rint(mean), 1, 65535)).astype("uint16"), index)
                dst.set_band_description(index, f"{name}_{t}")
            missing.append(float(no_data.mean()))
        dst.update_tags(**tags)
    tmp_path.replace(out_path)
    return missing


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def plan(paths: dict, cfg: dict) -> pd.DataFrame:
    """
    PS scenes available per ROI, reference date and window.

    A reference date is composited only when every window of every ROI has at
    least one PS scene, so both sensors and both ROIs share the same dates.
    """
    rois = load_rois(paths)
    ps = find_ps_scenes(paths)
    rows = []
    for ref in reference_dates(cfg["first_year"], cfg["last_year"]):
        for roi, geom in rois.geometry.items():
            for t, (start, end) in enumerate(window_ranges(ref, cfg["windows"])):
                n = len(scenes_in_window(ps, start, end, geom.bounds))
                rows.append({"reference_date": ref, "roi": roi, "window": t,
                             "start": start, "end": end, "PS_scenes": n})
    table = pd.DataFrame(rows)
    complete = table.groupby("reference_date")["PS_scenes"].min() > 0
    table["complete"] = table["reference_date"].map(complete)
    return table


def build_all(paths: dict, cfg: dict, overwrite: bool = False) -> pd.DataFrame:
    """Build every PS and S2 composite of the reference dates with PS scenes."""
    table = plan(paths, cfg)
    dates = sorted(table.loc[table["complete"], "reference_date"].unique())
    skipped = sorted(set(table["reference_date"]) - set(dates))
    print(f"    Reference dates with PS scenes for every ROI and window: "
          f"{', '.join(d.isoformat() for d in dates) or 'none'}")
    if skipped:
        print(f"    Skipped (missing PS scenes): {', '.join(d.isoformat() for d in skipped)}")
    if not dates:
        raise RuntimeError(f"No PlanetScope scenes ({PS_PATTERN}) found in {paths['PS_raw']}")

    rois = load_rois(paths)
    _ensure_s2_scenes(paths, cfg, rois, dates)

    ps = find_ps_scenes(paths)
    report = []
    for ref in dates:
        windows = window_ranges(ref, cfg["windows"])
        for roi, geom in rois.geometry.items():
            s2 = find_s2_scenes(paths, roi)
            for sensor, scenes in (("PS", ps), ("S2", s2)):
                out_path = composite_path(paths, sensor, roi, ref)
                by_window = [scenes_in_window(scenes, start, end, geom.bounds) for start, end in windows]
                if out_path.exists() and not overwrite:
                    print(f"    [OK] {out_path.name} exists")
                    continue
                if any(len(w) == 0 for w in by_window):
                    raise RuntimeError(f"{sensor} {roi} {ref}: a window has no scenes "
                                       f"({[len(w) for w in by_window]})")
                grid = roi_grid(geom.bounds, SENSORS[sensor]["resolution"])
                tags = {
                    "SENSOR": sensor, "ROI": roi, "REFERENCE_DATE": ref.isoformat(),
                    "WINDOWS_DAYS": str(cfg["windows"]),
                    **{f"SCENES_T{t}": ";".join(p.name for p in w["path"]) for t, w in enumerate(by_window)},
                }
                missing = build_composite(by_window, sensor, grid, out_path, tags)
                print(f"    [OK] {out_path.name}: scenes per window {[len(w) for w in by_window]}, "
                      f"pixels without observation {[f'{m:.1%}' for m in missing]}")
                for t, w in enumerate(by_window):
                    report.append({"sensor": sensor, "roi": roi, "reference_date": ref.isoformat(),
                                   "window": t, "scenes": len(w), "days": w["date"].nunique(),
                                   "no_observation_share": missing[t]})
    return pd.DataFrame(report)


def _ensure_s2_scenes(paths: dict, cfg: dict, rois, dates: list) -> None:
    """Offer to download the S2 scenes when a ROI/window has none locally."""
    missing = []
    for ref in dates:
        for roi in rois.index:
            s2 = find_s2_scenes(paths, roi)
            for start, end in window_ranges(ref, cfg["windows"]):
                if len(scenes_in_window(s2, start, end)) == 0:
                    missing.append((roi, ref))
                    break
    if not missing:
        return
    listing = ", ".join(f"{roi} {ref.isoformat()}" for roi, ref in sorted(set(missing)))
    print(f"    Sentinel-2 scenes missing in {paths['S2_raw']} for: {listing}")
    if not _ask("    Download them from Google Earth Engine now?"):
        raise RuntimeError(
            "Sentinel-2 scenes are missing. Run `python -m src.download_s2_gee` from the code "
            "folder (needs a Google Earth Engine account), then run the preprocessing again."
        )
    from src.download_s2_gee import download_s2

    download_s2(paths, cfg, rois, sorted({ref for _, ref in missing}))


if __name__ == "__main__":
    from src.paths import setup_paths

    base = Path(__file__).resolve().parents[2]
    report = build_all(setup_paths(base=base), read_settings(base / "code" / "config.toml"))
    print(report.to_string(index=False))
