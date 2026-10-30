"""
visual_inspection.py
--------------------
Main Fig. 2: visual inspection of the classifications of S2 and PS in four windows.

The windows were chosen from the validation points (SVC rbf, all bands), each at a different location and
date: a PMF field where PS missed most validation points (A), a greenhouse area where both sensors classified
every validation point correctly (B), an industrial area classified as PGH (C) and a greenhouse area with
missed validation points (D). In C, the areas classified as PGH were checked on very high-resolution imagery
(industrial roofs: false positives); in B and D, the parts of the reference polygons that the sensors left
unclassified were checked and are not greenhouses (potential errors of the manual polygons).

Rows: (1) PlanetScope true colour (B6, B4, B2) of one time step; (2) pixels classified as PMF or PGH by both
sensors, by S2 only or by PS only; (3) the reference polygons of the date and the exclusion zones.
Classes: the saved SVC rbf models with all bands (results/Models), applied to each window with the run's
preprocessing (zeros replaced by the whole-image band means, then the saved scaler), which reproduces the
full-image maps. Needs both composites of each window's ROI and date (data/composites/) and the polygons.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio.raw
import rasterio
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from scipy import ndimage

from .data import _read
from .experiments import PS_BANDS_ALL, S2_BANDS_ALL

FIGURE = "Figure_2_visual_inspection.png"
TABLE = "visual_inspection_panels.csv"

# Windows of 900 m x 816 m (centres in EPSG:32723). t: time step of the background (0-2).
# checked_fp: areas classified outside the polygons were checked and are false positives.
# polygon_error: reference parts missed by the sensors were checked and are errors of the polygons.
PANELS = [
    dict(panel="A", roi="Sumare", date="2022-12-15", t=1, x=266275.445, y=7466681.450, checked_fp=False, polygon_error=False),
    dict(panel="B", roi="Holambra", date="2022-12-15", t=0, x=289674.007, y=7498441.261, checked_fp=False, polygon_error=True),
    dict(panel="C", roi="Sumare", date="2022-12-15", t=0, x=274575.445, y=7464981.450, checked_fp=True, polygon_error=False),
    dict(panel="D", roi="Holambra", date="2023-06-18", t=0, x=289272.926, y=7498234.092, checked_fp=False, polygon_error=True),
]
WIDTH_M, HEIGHT_M = 900, 816
PIXEL_HA = 9e-4  # 3 m PlanetScope pixel

PMF, PGH = 0, 1
NAMES = {PMF: "PMF", PGH: "PGH"}
# Okabe-Ito palette (colour-blind safe), one hue per category; PMF "PS only" in blue-violet (the Okabe-Ito green
# looks like vegetation).
COLOURS = {("both", PMF): "#0072B2", ("S2", PMF): "#56B4E9", ("PS", PMF): "#7B4FD6",
           ("both", PGH): "#D55E00", ("S2", PGH): "#F0E442", ("PS", PGH): "#CC79A7"}
# Reference polygons in neutral colours absent from the class palette.
REF_COLOURS = {PMF: "white", PGH: "black"}
REF_HALO = {PMF: "black", PGH: "white"}
# Ellipses: line colour = sensor (with a halo), dashed = false positive, solid = false negative; each ellipse is
# labelled ("FP S2", "FN PS", ...). Red dashed: classified outside the polygons (not checked). Cyan: potential
# error of a reference polygon (bottom row).
SENSOR_LINE = {"S2": ("black", "white"), "PS": ("white", "black")}
OUTSIDE_LINE = ("#E8000B", "white")
POLYGON_ERROR_LINE = ("#00E5FF", "black")
DASHED, SOLID = (0, (5, 3)), "-"
MIN_HA = {"FP": 0.27, "FN": 0.05}  # smallest cluster worth an ellipse
TOP_N_CHECKED = 3  # false-positive clusters (either sensor) marked in panels with checked_fp


def _composite(paths, sensor, roi, date):
    return paths[f"{sensor}_composites"] / f"{sensor}_{roi}_{date}.tif"


def _classify(paths, sensor, roi, date, window):
    """Classes of a window with the saved SVC rbf pipeline, preprocessed as in the run."""
    columns = PS_BANDS_ALL if sensor == "PS" else S2_BANDS_ALL
    with rasterio.open(_composite(paths, sensor, roi, date)) as src:
        bands = list(src.descriptions)
        means = []
        for c in columns:  # SimpleImputer(missing_values=0, strategy="mean") fitted on the whole image
            a = src.read(bands.index(c) + 1)
            means.append(a[a != 0].astype(np.float64).mean())
        x = src.read([bands.index(c) + 1 for c in columns], window=window).astype(np.float64)
    h, w = x.shape[1:]
    x = np.where(x.reshape(len(columns), -1).T == 0, np.array(means), x.reshape(len(columns), -1).T)
    pipe = joblib.load(paths["models_path"] / f"{sensor}_2_SVC_rbf.joblib")
    return pipe.predict(x).reshape(h, w)


def _classes(paths, sensor, roi, date, transform, shape):
    """Classes on the PS panel grid (S2 resampled by nearest neighbour)."""
    west, south, east, north = rasterio.transform.array_bounds(shape[0], shape[1], transform)
    with rasterio.open(_composite(paths, sensor, roi, date)) as src:
        win = from_bounds(west, south, east, north, transform=src.transform).round_offsets().round_lengths()
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
        src_transform, crs = src.window_transform(win), src.crs
    labels = _classify(paths, sensor, roi, date, win)
    if sensor == "PS":
        return labels[1:-1, 1:-1][:shape[0], :shape[1]]
    dst = np.full(shape, 255, dtype=np.uint8)
    reproject(labels.astype(np.uint8), dst, src_transform=src_transform, src_crs=crs, dst_transform=transform,
              dst_crs=crs, resampling=Resampling.nearest)
    return dst


def _true_colour(paths, roi, date, t, window):
    with rasterio.open(_composite(paths, "PS", roi, date)) as src:
        bands = list(src.descriptions)
        rgb = src.read([bands.index(f"B{b}_{t}") + 1 for b in (6, 4, 2)], window=window).astype(float)
    lo, hi = np.percentile(rgb[rgb > 0], (1, 99))
    return (np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8).transpose(1, 2, 0)


def _layer(paths, name):
    """Geometries (shapely) and attribute fields of one GeoPackage layer."""
    meta, _, geometry, fields = pyogrio.raw.read(paths["polygons_path"] / f"{name}.gpkg")
    return shapely.from_wkb(geometry), dict(zip(meta["fields"], fields))


def _polygons(paths, date):
    """Reference polygons of the date: PMF of that reference date, PMF exclusion zones whose season covers it,
    and the (undated) PGH polygons and exclusion zones."""
    pmf, f = _layer(paths, "PMF_homogeneous")
    pmf = pmf[np.array([str(d)[:10] == date for d in f["reference_date"]])]
    ex, f = _layer(paths, "PMF_exclusion")
    covers = np.array([str(a)[:10] <= date <= str(b)[:10] for a, b in zip(f["season_start"], f["season_end"])])
    return {"PMF": pmf, "PMF exclusion": ex[covers], "PGH": _layer(paths, "PGH_homogeneous")[0],
            "PGH exclusion": _layer(paths, "PGH_exclusion")[0]}


def _mask(geoms, transform, shape):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(geoms, out_shape=shape, transform=transform).astype(bool)


def _clusters(m, kind, top_n):
    """Largest clusters of mask m (up to top_n, each at least MIN_HA[kind]), as (mask, hectares)."""
    if kind == "FN":  # reference polygons are small and separate: group misses less than 30 m apart
        lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=5))
        lab = np.where(m, lab, 0)
    else:  # compact clusters; the opening removes thin lines (roads), which would join separate roofs
        m = ndimage.binary_opening(m, iterations=2)
        lab, n = ndimage.label(m)
    if n == 0:
        return []
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    order = [k for k in np.argsort(sizes)[::-1][:top_n] if sizes[k] * PIXEL_HA >= MIN_HA[kind]]
    return [(lab == k + 1, round(float(sizes[k] * PIXEL_HA), 2)) for k in order]


def _ellipse(ax, comp, colour, halo, style, labels):
    """Ellipse (2 standard deviations) around one cluster; labels: [(text, side, colour, halo)]. colour None draws
    interleaved black (S2) and white (PS) dashes for a cluster of both sensors."""
    rows, cols = np.nonzero(comp)
    vals, vecs = np.linalg.eigh(np.cov(np.vstack([cols, rows])))
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * 2.0 * np.sqrt(np.maximum(vals[::-1], 1)) + 12
    cx, cy = cols.mean(), rows.mean()
    if colour is None:
        for c, phase in (("black", 0), ("white", 5)):
            ax.add_patch(Ellipse((cx, cy), w, h, angle=angle, fill=False, edgecolor=c, linewidth=2.8,
                                 linestyle=(phase, (5, 5))))
    else:
        e = Ellipse((cx, cy), w, h, angle=angle, fill=False, edgecolor=colour, linestyle=style, linewidth=2.2)
        e.set_path_effects([pe.withStroke(linewidth=4.2, foreground=halo)])
        ax.add_patch(e)
    t, a = np.linspace(0, 2 * np.pi, 360), np.radians(angle)
    xs = cx + w / 2 * np.cos(t) * np.cos(a) - h / 2 * np.sin(t) * np.sin(a)
    ys = cy + w / 2 * np.cos(t) * np.sin(a) + h / 2 * np.sin(t) * np.cos(a)
    inside = (xs > 8) & (xs < comp.shape[1] - 8) & (ys > 8) & (ys < comp.shape[0] - 8)
    for text, side, c, hl in labels:  # on the ellipse, at its leftmost or rightmost point inside the panel
        pick = np.argmin if side == "left" else np.argmax
        i = np.flatnonzero(inside)[pick(xs[inside])] if inside.any() else int(pick(xs))
        ax.text(xs[i], ys[i], text, color=c, fontsize=10, fontweight="bold", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground=hl)])


def _sensor_label(text, sensor):
    return (text, "left" if sensor == "S2" else "right", *SENSOR_LINE[sensor])


def _validation_points(paths, results_path):
    """Validation points with the SVC rbf (all bands) prediction of each sensor."""
    v = _read(paths["datasets_path"] / "PS_Validation.csv")[["Class", "X", "Y", "Reference_Date"]]
    p = pd.read_csv(results_path / "validation_predictions.csv.gz")
    p = p[(p["Models"] == "SVC_rbf") & p["Imagery"].isin(["S2_2", "PS_2"])]
    return v.join(p.pivot(index="Point", columns="Imagery", values="Predicted Class").rename(
        columns={"S2_2": "S2", "PS_2": "PS"}))


def plot_visual_inspection(paths: dict, results_path: Path, figures_path: Path) -> None:
    """Draw main Fig. 2 and write the panel numbers given in the SM text (results/visual_inspection_panels.csv)."""
    needed = [_composite(paths, s, p["roi"], p["date"]) for p in PANELS for s in ("PS", "S2")]
    needed += [paths["models_path"] / f"{s}_2_SVC_rbf.joblib" for s in ("PS", "S2")]
    missing = sorted({str(f) for f in needed if not Path(f).exists()})
    if missing:
        raise FileNotFoundError(f"needs {len(missing)} more input file(s), e.g. {missing[0]}")
    points = _validation_points(paths, results_path)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}):
        fig, axes = plt.subplots(3, len(PANELS), figsize=(4 * len(PANELS), 10.6),
                                 gridspec_kw={"wspace": 0.02, "hspace": 0.03}, squeeze=False)
        rows_out, drawn = [], set()
        for col, p in enumerate(PANELS):
            with rasterio.open(_composite(paths, "PS", p["roi"], p["date"])) as src:
                win = from_bounds(p["x"] - WIDTH_M / 2, p["y"] - HEIGHT_M / 2, p["x"] + WIDTH_M / 2, p["y"] + HEIGHT_M / 2,
                                  transform=src.transform).round_offsets().round_lengths()
                transform = src.window_transform(win)
            shape = (int(win.height), int(win.width))
            rgb = _true_colour(paths, p["roi"], p["date"], p["t"], win)
            lab = {s: _classes(paths, s, p["roi"], p["date"], transform, shape) for s in ("PS", "S2")}
            poly = _polygons(paths, p["date"])
            ref = {k: _mask(poly[NAMES[k]], transform, shape) for k in (PMF, PGH)}
            exc = {k: _mask(poly[f"{NAMES[k]} exclusion"], transform, shape) for k in (PMF, PGH)}
            known = ref[PMF] | ref[PGH] | exc[PMF] | exc[PGH]

            for row in range(3):
                axes[row, col].imshow(rgb, interpolation="nearest")
                axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
            over = np.zeros(shape + (4,))
            for k in (PMF, PGH):
                s2, ps = lab["S2"] == k, lab["PS"] == k
                over[s2 & ~ps] = matplotlib.colors.to_rgba(COLOURS[("S2", k)])
                over[ps & ~s2] = matplotlib.colors.to_rgba(COLOURS[("PS", k)])
                over[ps & s2] = matplotlib.colors.to_rgba(COLOURS[("both", k)])
            axes[1, col].imshow(over, interpolation="nearest")
            for k in (PMF, PGH):
                fill = np.zeros(shape + (4,))
                fill[ref[k]] = matplotlib.colors.to_rgba(REF_COLOURS[k], 0.45)
                axes[2, col].imshow(fill, interpolation="nearest")
                if exc[k].any():
                    axes[2, col].contour(exc[k], levels=[0.5], colors=["#BBBBBB"], linewidths=1.3, linestyles="dotted")
                if ref[k].any():
                    cs = axes[2, col].contour(ref[k], levels=[0.5], colors=[REF_COLOURS[k]], linewidths=1.2)
                    cs.set_path_effects([pe.withStroke(linewidth=2.4, foreground=REF_HALO[k])])
            axes[0, col].text(0.03, 0.97, p["panel"], transform=axes[0, col].transAxes, va="top", ha="left",
                              fontsize=16, bbox=dict(facecolor="white", edgecolor="black"))

            # Validation points of the window and its date
            west, south, east, north = rasterio.transform.array_bounds(shape[0], shape[1], transform)
            v = points[(points["Reference_Date"] == p["date"]) & points["X"].between(west, east)
                       & points["Y"].between(south, north)]
            r = {"Panel": p["panel"], "ROI": p["roi"], "Reference Date": p["date"], "Background Time Step": p["t"],
                 "Centre X": p["x"], "Centre Y": p["y"], "Width (m)": WIDTH_M, "Height (m)": HEIGHT_M,
                 "Validation APS Points": int((v["Class"] < 2).sum()),
                 "Validation Others Points": int((v["Class"] == 2).sum())}
            for s in ("S2", "PS"):
                r[f"Missed {s}"] = int(((v["Class"] < 2) & (v[s] == 2)).sum())
                r[f"Wrongly Classified {s}"] = int(((v["Class"] == 2) & (v[s] < 2)).sum())

            fn_all = np.zeros(shape, bool)
            fp_all = {k: np.zeros(shape, bool) for k in (PMF, PGH)}
            fp_by = {}
            for k in (PMF, PGH):
                for s in ("S2", "PS"):
                    pred = lab[s] == k
                    fp, fn = pred & ~known, ref[k] & ~pred
                    r[f"{NAMES[k]} Outside Polygons {s} (ha)"] = round(float(fp.sum() * PIXEL_HA), 2)
                    if p["checked_fp"]:  # marked below, from the clusters of both sensors
                        fp_all[k] |= fp
                        fp_by[(k, s)] = fp
                    else:
                        for comp, _ in _clusters(fp, "FP", 1):
                            _ellipse(axes[1, col], comp, *OUTSIDE_LINE, DASHED, [(s, "left" if s == "S2" else "right", *OUTSIDE_LINE)])
                            drawn.add("outside")
                    if p["polygon_error"]:
                        fn_all |= fn
                    else:
                        for comp, _ in _clusters(fn, "FN", 1):
                            _ellipse(axes[1, col], comp, *SENSOR_LINE[s], SOLID, [_sensor_label(f"FN {s}", s)])
            if p["checked_fp"]:  # a sensor counts for a cluster if it has at least 25 % of it
                for k in (PMF, PGH):
                    for comp, _ in _clusters(fp_all[k], "FP", TOP_N_CHECKED):
                        by = [s for s in ("S2", "PS") if (fp_by[(k, s)] & comp).sum() >= 0.25 * comp.sum()]
                        if len(by) == 2:
                            _ellipse(axes[1, col], comp, None, None, DASHED,
                                     [_sensor_label("FP S2", "S2"), _sensor_label("FP PS", "PS")])
                        else:
                            _ellipse(axes[1, col], comp, *SENSOR_LINE[by[0]], DASHED, [_sensor_label(f"FP {by[0]}", by[0])])
            if p["polygon_error"]:
                found = _clusters(fn_all, "FN", 3)
                for comp, _ in found:
                    _ellipse(axes[2, col], comp, *POLYGON_ERROR_LINE, SOLID, [])
                    drawn.add("polygon error")
                # validation points missed by each sensor inside the potential polygon errors (30 m tolerance)
                near = ndimage.binary_dilation(np.any([c for c, _ in found], axis=0), iterations=10) if found else fn_all
                rr, cc = rasterio.transform.rowcol(transform, v["X"].values, v["Y"].values)
                rr = np.clip(np.asarray(rr), 0, shape[0] - 1)
                cc = np.clip(np.asarray(cc), 0, shape[1] - 1)
                for s in ("S2", "PS"):
                    r[f"Missed {s} in Polygon Errors"] = int((((v["Class"] < 2) & (v[s] == 2)).values & near[rr, cc]).sum())
            for row in range(3):  # ellipses must not rescale the panels
                axes[row, col].set_xlim(-0.5, shape[1] - 0.5)
                axes[row, col].set_ylim(shape[0] - 0.5, -0.5)
            rows_out.append(r)

        # Legend in fixed columns: PMF classes | PGH classes | reference | other marks (FP/FN are labelled on the
        # ellipses).
        blank = Line2D([], [], alpha=0, label=" ")
        label = {"both": "both sensors", "S2": "S2 only", "PS": "PS only"}
        cols = [[Patch(color=COLOURS[(s, k)], label=f"{NAMES[k]}: {label[s]}") for s in ("both", "S2", "PS")]
                for k in (PMF, PGH)]
        cols.append([Patch(facecolor=matplotlib.colors.to_rgba(REF_COLOURS[k], 0.45),
                           edgecolor=REF_HALO[k] if k == PMF else REF_COLOURS[k], label=f"{NAMES[k]} reference polygon")
                     for k in (PMF, PGH)]
                    + [Line2D([], [], color="#BBBBBB", linestyle="dotted", linewidth=1.8, label="exclusion zone")])
        marks = []
        for key, (colour, halo), style, text in (("polygon error", POLYGON_ERROR_LINE, SOLID, "potential reference polygon error"),
                                                 ("outside", OUTSIDE_LINE, DASHED, "classified outside polygons")):
            if key in drawn:
                marks.append(Line2D([], [], color=colour, linestyle=style, linewidth=2.2,
                                    path_effects=[pe.withStroke(linewidth=4.2, foreground=halo)], label=text))
        if marks:
            cols.append(marks)
        n_rows = max(len(c) for c in cols)
        handles = [h for c in cols for h in c + [blank] * (n_rows - len(c))]
        fig.subplots_adjust(bottom=0.075)
        fig.legend(handles=handles, loc="lower center", ncol=len(cols), frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, 0.0), handlelength=3.2, columnspacing=1.2)
        fig.savefig(figures_path / FIGURE, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    pd.DataFrame(rows_out).to_csv(results_path / TABLE, index=False)
