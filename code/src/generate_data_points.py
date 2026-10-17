"""
generate_data_points.py

Draw the training and validation points from the reference polygons and
extract their PlanetScope (PS) and Sentinel-2 (S2) values from the
3-time-series composites, rewriting

    data/PS_Training.csv, data/PS_Validation.csv,
    data/S2_Training.csv, data/S2_Validation.csv

with the columns Class, X, Y, Reference_Date and the bands <band>_<t>.

Classes: 0 = plastic-mulched farmland (PMF), 1 = plastic greenhouses (PGH),
2 = others.

Sampling rules
- Sets: each ROI is split in two halves (data/polygons/ROIs.gpkg); training
  points come from the training halves and validation points from the
  validation halves.
- Reference dates: the PMF reference dates with PS and S2 composites for
  every ROI.
- One point per S2 pixel and reference date. A 10 m S2 pixel is eligible when
  its centre lies inside the class region and it has a value in every band of
  both sensors; the point is placed at random inside the pixel (and inside the
  class region), and its PS values come from the 3 m pixel at the point.
- PMF: polygons of the same reference date. PGH: all polygons (greenhouses
  have no date; a location may appear at several reference dates, as a
  different observation). Others: outside the PMF and PGH exclusion zones and
  homogeneous polygons of every year; a location is used only once.
- Balance: the same number of points per class (config.toml). PMF and PGH
  points are split equally across reference dates; a date with fewer eligible
  pixels than its share gives the rest to the other dates. Others are split
  equally between the ROIs and then across dates.
- The same points are used for both sensors.

Run from the code folder:  python -m src.generate_data_points
"""

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows

from src.config import DEFAULT_RANDOM_STATE
from src.preprocessing import GRID_CRS, SENSORS, composite_path, read_settings

CLASSES = {"PMF": 0, "PGH": 1, "others": 2}
SETS = {"training": "Training", "validation": "Validation"}
MAX_TRIES = 50  # random positions tried inside a pixel before it is dropped


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_polygons(paths: dict) -> dict:
    import geopandas as gpd

    folder = paths["polygons_path"]
    layers = ["ROIs", "PMF_homogeneous", "PMF_exclusion", "PGH_homogeneous", "PGH_exclusion"]
    return {name: gpd.read_file(folder / f"{name}.gpkg").to_crs(GRID_CRS) for name in layers}


def available_dates(paths: dict, rois: list) -> list:
    """Reference dates with PS and S2 composites for every ROI."""
    found = {}
    for sensor in SENSORS:
        for path in paths[f"{sensor}_composites"].glob(f"{sensor}_*_*.tif"):
            match = re.fullmatch(rf"{sensor}_(.+)_(\d{{4}}-\d{{2}}-\d{{2}})\.tif", path.name)
            if match:
                found.setdefault(dt.date.fromisoformat(match.group(2)), set()).add((sensor, match.group(1)))
    needed = {(sensor, roi) for sensor in SENSORS for roi in rois}
    return sorted(date for date, have in found.items() if needed <= have)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

def allocate(target: int, capacity: dict) -> dict:
    """
    Split *target* points equally across the keys of *capacity*, never above a
    key's capacity; what a key cannot take goes to the keys with room left.
    """
    keys = sorted(capacity)
    counts = {key: 0 for key in keys}
    remaining = target
    open_keys = [key for key in keys if capacity[key] > 0]
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        for i, key in enumerate(open_keys):
            take = min(share + (i < extra), capacity[key] - counts[key])
            counts[key] += take
            remaining -= take
        open_keys = [key for key in open_keys if counts[key] < capacity[key]]
    if remaining > 0:
        table = "\n".join(f"  {key}: capacity {capacity[key]}" for key in keys)
        raise ValueError(f"Only {target - remaining} of {target} points fit:\n{table}")
    return counts


# ---------------------------------------------------------------------------
# Candidate pixels and points
# ---------------------------------------------------------------------------

STRIP = 512  # rows read at once (all bands), efficient for any interleaving


def _strips(src):
    for row in range(0, src.height, STRIP):
        window = rasterio.windows.Window(0, row, src.width, min(STRIP, src.height - row))
        yield row, src.read(window=window)


class Composite:
    """A composite's grid and valid-pixel mask (a value in every band)."""

    def __init__(self, path: Path):
        self.path = path
        with rasterio.open(path) as src:
            self.transform = src.transform
            self.shape = (src.height, src.width)
            self.names = list(src.descriptions)
            self.valid = np.zeros(self.shape, dtype=bool)
            for row, data in _strips(src):
                self.valid[row:row + data.shape[1]] = (data != src.nodata).all(axis=0)

    def index(self, x, y):
        cols, rows = ~self.transform * (np.asarray(x), np.asarray(y))
        return np.floor(rows).astype(int), np.floor(cols).astype(int)

    def is_valid(self, x, y):
        rows, cols = self.index(x, y)
        inside = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        out = np.zeros(len(rows), dtype=bool)
        out[inside] = self.valid[rows[inside], cols[inside]]
        return out

    def values(self, x, y) -> pd.DataFrame:
        rows, cols = self.index(x, y)
        with rasterio.open(self.path) as src:
            out = np.zeros((len(rows), len(self.names)), dtype=src.dtypes[0])
            for row, data in _strips(src):
                here = (rows >= row) & (rows < row + data.shape[1])
                out[here] = data[:, rows[here] - row, cols[here]].T
        return pd.DataFrame(out, columns=self.names)


def _pixel_centres(s2: Composite, half) -> tuple:
    """Centres (x, y), rows and cols of the S2 pixels whose centre lies in *half*."""
    import shapely

    t = s2.transform
    cols = np.arange(s2.shape[1])
    rows = np.arange(s2.shape[0])
    xs = t.c + (cols + 0.5) * t.a
    ys = t.f + (rows + 0.5) * t.e
    left, bottom, right, top = half.bounds
    cols = cols[(xs >= left) & (xs < right)]
    rows = rows[(ys >= bottom) & (ys < top)]
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    x = t.c + (cc + 0.5) * t.a
    y = t.f + (rr + 0.5) * t.e
    keep = shapely.contains_xy(half, x, y)
    return x[keep], y[keep], rr[keep], cc[keep]


def _in_region(region, inside: bool, x, y):
    import shapely

    hit = shapely.contains_xy(region, x, y)
    return hit if inside else ~hit


def _random_points(x, y, half_size, region, inside, rng):
    """A random point inside each pixel and inside (or outside) *region*; NaN if none found."""
    px = np.full(len(x), np.nan)
    py = np.full(len(y), np.nan)
    todo = np.arange(len(x))
    for _ in range(MAX_TRIES):
        if len(todo) == 0:
            break
        cx = x[todo] + rng.uniform(-half_size, half_size, len(todo))
        cy = y[todo] + rng.uniform(-half_size, half_size, len(todo))
        ok = _in_region(region, inside, cx, cy)
        px[todo[ok]], py[todo[ok]] = cx[ok], cy[ok]
        todo = todo[~ok]
    return px, py


def candidates(s2: Composite, ps: Composite, half, region, inside: bool, rng) -> pd.DataFrame:
    """Eligible S2 pixels of *half* for a class region, with one random point each."""
    import shapely

    shapely.prepare(region)
    x, y, rows, cols = _pixel_centres(s2, half)
    keep = _in_region(region, inside, x, y)
    x, y, rows, cols = x[keep], y[keep], rows[keep], cols[keep]
    keep = s2.valid[rows, cols]
    x, y, rows, cols = x[keep], y[keep], rows[keep], cols[keep]
    px, py = _random_points(x, y, abs(s2.transform.a) / 2, region, inside, rng)
    keep = ~np.isnan(px)
    keep[keep] = ps.is_valid(px[keep], py[keep])
    return pd.DataFrame({"row": rows[keep], "col": cols[keep], "X": px[keep], "Y": py[keep]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(paths: dict, cfg: dict, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Write the four point CSVs; return the summary (capacity and points per group)."""
    rng = np.random.default_rng(random_state)
    layers = load_polygons(paths)
    roi_names = sorted(layers["ROIs"]["roi"].unique())
    pmf = layers["PMF_homogeneous"]
    pmf_dates = {d.date() for d in pd.to_datetime(pmf["reference_date"])}
    dates = [d for d in available_dates(paths, roi_names) if d in pmf_dates]
    if not dates:
        raise RuntimeError("No PMF reference date has PS and S2 composites for every ROI.")
    print(f"    Reference dates: {', '.join(d.isoformat() for d in dates)}")

    regions = {
        "PGH": (layers["PGH_homogeneous"].union_all(), True),
        "others": (pd.concat([layers[name].geometry for name in
                              ("PMF_exclusion", "PGH_exclusion", "PMF_homogeneous", "PGH_homogeneous")])
                   .union_all(), False),
    }
    for date in dates:
        regions[("PMF", date)] = (pmf[pd.to_datetime(pmf["reference_date"]).dt.date == date].union_all(), True)

    targets = {"training": cfg["training_points"], "validation": cfg["validation_points"]}
    summary, tables = [], {s: [] for s in SETS}
    composites = {
        (sensor, roi, date): Composite(composite_path(paths, sensor, roi, date))
        for sensor in SENSORS for roi in roi_names for date in dates
    }
    rois = layers["ROIs"]

    for set_name, target in targets.items():
        # Eligible pixels per class, ROI and date (with one random point each)
        pool = {}
        for roi in roi_names:
            half = rois[(rois["roi"] == roi) & (rois["set"] == set_name)].union_all()
            for date in dates:
                s2, ps = composites["S2", roi, date], composites["PS", roi, date]
                for cls in CLASSES:
                    region, inside = regions[("PMF", date)] if cls == "PMF" else regions[cls]
                    pool[cls, roi, date] = candidates(s2, ps, half, region, inside, rng)

        for cls in CLASSES:
            if cls == "others":
                per_roi = allocate(target, {roi: 10**9 for roi in roi_names})
                plan = {}
                for roi in roi_names:
                    capacity = {date: len(pool[cls, roi, date]) for date in dates}
                    for date, n in allocate(per_roi[roi], capacity).items():
                        plan[roi, date] = n
            else:
                capacity = {date: sum(len(pool[cls, roi, date]) for roi in roi_names) for date in dates}
                per_date = allocate(target, capacity)
                plan = {}
                for date in dates:
                    by_roi = {roi: len(pool[cls, roi, date]) for roi in roi_names}
                    plan.update({(roi, date): n for roi, n in _proportional(per_date[date], by_roi).items()})

            used = {roi: set() for roi in roi_names}
            for (roi, date), n in sorted(plan.items(), key=lambda item: (item[0][1], item[0][0])):
                cand = pool[cls, roi, date]
                if cls == "others":  # a location is used only once across dates
                    taken = pd.Series(list(zip(cand["row"], cand["col"]))).isin(used[roi]).to_numpy()
                    cand = cand[~taken]
                if n > len(cand):
                    raise ValueError(f"{cls} {roi} {date}: {n} points requested, {len(cand)} pixels left")
                chosen = cand.iloc[rng.choice(len(cand), size=n, replace=False)] if n else cand.iloc[:0]
                used[roi].update(zip(chosen["row"], chosen["col"]))
                tables[set_name].append(chosen.assign(Class=CLASSES[cls], ROI=roi, Reference_Date=date))
                summary.append({"set": set_name, "class": cls, "roi": roi, "reference_date": date.isoformat(),
                                "eligible_pixels": len(pool[cls, roi, date]), "points": n})

    _write(tables, composites, paths)
    return pd.DataFrame(summary)


def _proportional(total: int, capacity: dict) -> dict:
    """Split *total* across ROIs in proportion to their eligible pixels (one plastic class per ROI here)."""
    available = sum(capacity.values())
    if total == 0 or available == 0:
        return {roi: 0 for roi in capacity}
    counts = {roi: int(total * n // available) for roi, n in capacity.items()}
    for roi in sorted(capacity, key=lambda r: -capacity[r])[: total - sum(counts.values())]:
        counts[roi] += 1
    return counts


def _write(tables: dict, composites: dict, paths: dict) -> None:
    for set_name, parts in tables.items():
        points = (pd.concat(parts, ignore_index=True)
                  .sort_values(["Reference_Date", "Class"], kind="stable")
                  .reset_index(drop=True))
        for sensor in SENSORS:
            frames = []
            for (roi, date), group in points.groupby(["ROI", "Reference_Date"], sort=False):
                values = composites[sensor, roi, date].values(group["X"], group["Y"])
                frames.append(pd.concat([group[["Class", "X", "Y", "Reference_Date"]].reset_index(), values], axis=1))
            table = pd.concat(frames).set_index("index").sort_index()
            table["Reference_Date"] = [d.isoformat() for d in table["Reference_Date"]]
            assert not table.isna().any().any(), f"{sensor} {set_name}: missing values"
            out = paths["datasets_path"] / f"{sensor}_{SETS[set_name]}.csv"
            table.round({"X": 3, "Y": 3}).to_csv(out, index=False)
            print(f"    [OK] {out.name}: {len(table)} points")


if __name__ == "__main__":
    from src.paths import setup_paths

    base = Path(__file__).resolve().parents[2]
    paths = setup_paths(base=base)
    summary = generate(paths, read_settings(base / "code" / "config.toml"))
    out = paths["scratch_path"] / "data_points_summary.csv"
    summary.to_csv(out, index=False)
    print(summary.groupby(["set", "class", "reference_date"])[["eligible_pixels", "points"]].sum().to_string())
