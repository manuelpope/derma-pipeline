"""Lesion metrics: tabular reporting and circularity.

Replaces notebook cells 25 and 27, which used ``display(df)`` (a Colab/
IPython-only function) and never persisted the data.

Stage 8/9/10 metrics (color + asymmetry + radial) are gated behind the
optional ``rgb`` argument: pass the source image to enable them; the base
contour metrics (area, perimeter, circularity) work without it.

Stage 11/12/13/14 (dermoscopic palette, diameter, 2-axis asymmetry, TDS)
also require ``rgb``; their columns are appended to the per-lesion row
when ``rgb`` is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from derm_pipeline.borders import (
    asymmetry_score,
    asymmetry_score_2axis,
    border_irregularity,
    equivalent_diameter,
    radial_profile,
)


# Canonical dermoscopic color palette (Zalaudek, Argenziano, Soyer).
# Tuples are LAB thresholds derived from standard dermoscopy literature.
# The names match what dermatologists use in clinical reports.
DERMO_COLOR_NAMES: tuple[str, ...] = (
    "white",
    "red",
    "light_brown",
    "dark_brown",
    "blue_gray",
    "black",
)


def lesions_to_df(lesions: Iterable[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of lesion descriptors."""
    return pd.DataFrame(list(lesions))


def _mean_color(
    color_img: NDArray[np.uint8],
    mask_uint8: NDArray[np.uint8],
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Channel-wise mean of ``color_img`` over the pixels where ``mask > 0``.

    OpenCV's ``cv2.mean`` returns ``(b, g, r, a)`` — we strip the alpha channel
    for 3-channel inputs. Returns ``(None, None, None)`` for an empty mask.
    """
    if int(np.sum(mask_uint8 > 0)) == 0:
        return (None, None, None)
    means = cv2.mean(color_img, mask=mask_uint8)
    return float(means[0]), float(means[1]), float(means[2])


def classify_dermo_colors(lab_pixels: NDArray[np.float32]) -> dict[str, float]:
    """Classify LAB pixels into the 6 canonical dermoscopic colors.

    Returns a dict mapping each color name in :data:`DERMO_COLOR_NAMES` to
    the fraction of ``lab_pixels`` that fall into that class (values sum to
    1.0, except for empty input which returns zeros).

    LAB thresholds follow the standard dermoscopy literature; ``L ∈ [0, 255]``
    (OpenCV's scaled range, not the perceptually uniform [0, 100]). Pixels
    not matching any class are left uncolored (the fractions will sum to
    less than 1.0). In practice ≥95% of skin-lesion pixels fall into one of
    these six categories.

    Classification is vectorized with ``np.where`` — O(N) regardless of how
    many pixels are passed.
    """
    result: dict[str, float] = {name: 0.0 for name in DERMO_COLOR_NAMES}
    if lab_pixels is None or len(lab_pixels) == 0:
        result["n_colors_present"] = 0
        return result

    L = lab_pixels[:, 0]
    a = lab_pixels[:, 1]
    b = lab_pixels[:, 2]

    # Order matters: check the most restrictive classes first (white, black)
    # before the broader brown family. The chosen thresholds intentionally
    # overlap on edges — np.where picks the first match, so each pixel ends
    # up in exactly one bucket.
    cls = np.full(len(lab_pixels), -1, dtype=np.int8)

    # white: very bright, neutral to slightly warm
    cls = np.where((L > 200) & (a < 135) & (b < 140), 0, cls)
    # black: very dark
    cls = np.where((cls == -1) & (L < 85), 5, cls)
    # red: high a*, dark L
    cls = np.where((cls == -1) & (a > 165) & (L < 130) & (b < 150), 1, cls)
    # blue_gray: low L, distinctly positive b*
    cls = np.where((cls == -1) & (L < 160) & (b > 145), 4, cls)
    # dark_brown: mid-low L, mid-high a*/b*
    cls = np.where(
        (cls == -1) & (L >= 85) & (L < 160) & (a >= 130) & (a < 175) & (b >= 130) & (b < 170),
        3,
        cls,
    )
    # light_brown: bright L, mid a*/b* (catches everything else brown-ish)
    cls = np.where(
        (cls == -1) & (L >= 150) & (L < 220) & (a >= 125) & (a < 165) & (b >= 125) & (b < 170),
        2,
        cls,
    )

    n = len(lab_pixels)
    for idx, name in enumerate(DERMO_COLOR_NAMES):
        result[name] = float(np.sum(cls == idx) / n)

    # `n_colors_present` is a derived scalar — count the buckets with > 1 %
    # coverage (anything below that is noise).
    result["n_colors_present"] = int(sum(1 for v in result.values() if v > 0.01))
    return result


def total_dermoscopy_score(
    A: float,
    B: float,
    C: float,
    D: float,
) -> tuple[float, str]:
    """Total Dermoscopy Score (Nachbar et al., JAAD 1994).

    Computes::

        TDS = (A × 1.3) + (B × 0.1) + (C × 0.5) + (D × 0.5)

    Inputs in their native scales:
        A ∈ [0, 2] — 2-axis asymmetry score
        B ∈ [0, 8] — octant border score (we use a proxy derived from
                     ``border_irregularity``: ``B = min(8, round(irreg × 16))``)
        C ∈ [1, 6] — number of distinct named dermoscopic colors present
        D ∈ [1, 5] — number of distinct dermoscopic structures (proxy:
                     clamped to the count of named colors, capped at 5)

    Returns ``(tds, classification)`` where classification is one of:
        ``"benign"``         (TDS < 4.75)
        ``"suspicious"``     (4.75 ≤ TDS < 5.45)
        ``"highly_suspicious"`` (TDS ≥ 5.45)

    Note: this is a teaching/research tool, **not** a diagnostic device.
    The original TDS was derived on a specific population; values should be
    interpreted by a clinician in context.
    """
    A = float(np.clip(A, 0.0, 2.0))
    B = float(np.clip(B, 0.0, 8.0))
    C = float(np.clip(C, 1.0, 6.0))
    D = float(np.clip(D, 1.0, 5.0))
    tds = A * 1.3 + B * 0.1 + C * 0.5 + D * 0.5
    if tds < 4.75:
        cls = "benign"
    elif tds < 5.45:
        cls = "suspicious"
    else:
        cls = "highly_suspicious"
    return float(tds), cls


def diagnostic_flag(value: float, *, low: float, high: float) -> tuple[float, str]:
    """Map a clinical metric to a 0–10 score and a green/yellow/red flag.

    The score scales linearly in three zones:

        value < low       → score in [0, 3], flag = "green"
        low ≤ value < high → score in [3, 6], flag = "yellow"
        value ≥ high      → score in [6, 10], flag = "red"

    Args:
        value: The raw clinical metric (e.g. n_colors_present, d_eq_px,
               tds_score, octant score, …).
        low:  Threshold below which the metric is treated as benign.
        high: Threshold at/above which the metric is treated as highly
              suspicious.

    Returns:
        ``(score_0_10, flag_color)`` where ``flag_color`` ∈
        ``{"green", "yellow", "red"}``.
    """
    v = float(value)
    lo, hi = float(low), float(high)
    if hi <= lo:
        # Degenerate thresholds — treat as always suspicious to avoid div-by-zero.
        return 5.0, "yellow"
    if v < lo:
        # Linear from 0 (at value=0) up to 3 (at value=lo).
        score = 3.0 * (v / lo) if lo > 0 else 0.0
        return float(max(0.0, min(3.0, score))), "green"
    if v < hi:
        # Linear from 3 (at value=lo) up to 6 (at value=hi).
        score = 3.0 + 3.0 * ((v - lo) / (hi - lo))
        return float(max(3.0, min(6.0, score))), "yellow"
    # Above high — linear ramp from 6 to 10, then capped. The high value is
    # treated as a "soft saturation" point: one full threshold-width above
    # high = 10/10.
    score = 6.0 + 4.0 * ((v - hi) / (hi - lo))
    return float(max(6.0, min(10.0, score))), "red"


# Per-axis thresholds used to score the four ABCDE components. They are the
# values at which the green→yellow and yellow→red transitions happen.
AXIS_THRESHOLDS: dict[str, tuple[float, float]] = {
    # ABCD-A: 2-axis asymmetry score, range [0, 2].
    # > 1 means asymmetric on at least one full axis — clinical rule of thumb.
    "A": (0.5, 1.0),
    # ABCD-B: border octant score, integer [0, 8].
    # > 4 means ≥ 5 octants with abrupt cutoff — clearly irregular.
    "B": (2.0, 4.0),
    # ABCD-C: number of named dermoscopic colors present, [1, 6].
    # > 3 colors is variegated — strong melanoma sign.
    "C": (2.0, 4.0),
    # ABCD-D: equivalent-circular diameter in pixels. 600 px ≈ the
    # clinical "6 mm rule" at the resolution this pipeline targets.
    "D": (300.0, 600.0),
}


def _border_octant_score(border_irregularity_value: float) -> int:
    """Map ``border_irregularity`` (CV of radial distances) to ABCD B-scale.

    The ABCD rule scores border on a 0–8 octant scale (number of octants
    with abrupt pigment cutoff). We don't do real octant analysis in Tier 1
    — we map the radial-CV scalar linearly to the 0–8 range and clamp::

        B = min(8, round(border_irregularity × 16))

    So a perfect circle (irreg ≈ 0) → B = 0; very irregular borders
    (irreg ≥ 0.5) → B = 8. This is a coarse proxy, not a real octant
    analysis — flagged in the docstring of every function that consumes it.
    """
    return int(min(8, max(0, round(float(border_irregularity_value) * 16.0))))


def compute_metrics(
    lesions: list[dict],
    labels: NDArray[np.int32],
    rgb: NDArray[np.uint8] | None = None,
    *,
    large_diameter_px: float = 600.0,
) -> pd.DataFrame:
    """Build the per-lesion metrics table.

    Always-on columns (work without ``rgb``):

        id, x, y, width, height, area, cx, cy
        contour_area, perimeter, circularity
        radial_std          — std of radial distances from centroid (px)
        border_irregularity — std(radii) / mean(radii), dimensionless
        border_octant_score — ABCD-B proxy in [0, 8]
        symmetry            — IoU with horizontal flip around centroid

    Gated on ``rgb`` (mean color over each lesion's mask, on the **original**
    image, not the LAB-enhanced variant):

        mean_L, mean_a, mean_b            — LAB
        mean_H, mean_S, mean_V            — HSV
        asymmetry_h, asymmetry_v          — 2-axis asymmetry (Stage 13)
        asymmetry_2axis_score             — sum (0..2, ABCD A)
        diameter_eq_px, diameter_major_px — Stage 12
        diameter_large_flag               — True when d_eq ≥ large_diameter_px
        pct_white, pct_red, pct_light_brown, pct_dark_brown,
        pct_blue_gray, pct_black           — Stage 11 named-color fractions
        n_colors_present                  — count of buckets with > 1 % coverage
        tds_score, tds_class              — Stage 14 (Total Dermoscopy Score)

    Circularity uses the standard ``4πA / P²`` formula; a perfect circle
    scores ``1.0``. Perimeter ``0`` yields circularity ``0`` instead of NaN.

    Returns:
        DataFrame sorted by ``area`` descending. Empty if no lesions had
        contours.
    """
    # Precompute color transforms once so the per-lesion loop stays O(N).
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB) if rgb is not None else None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if rgb is not None else None

    rows: list[dict] = []
    for obj in lesions:
        obj_id = obj["id"]
        component_mask = np.uint8(labels == obj_id) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))

        if perimeter > 0:
            circularity = (4 * np.pi * contour_area) / (perimeter ** 2)
        else:
            circularity = 0.0

        # Radial profile → std(absolute) + CV(dimensionless)
        (rcx, rcy), radii, _ = radial_profile(component_mask)
        if radii.size > 0 and float(np.mean(radii)) > 0:
            radial_std = float(np.std(radii))
            irregularity = float(np.std(radii) / np.mean(radii))
        else:
            radial_std = 0.0
            irregularity = 0.0
        symmetry = asymmetry_score(component_mask)
        b_oct = _border_octant_score(irregularity)

        row: dict = {
            **obj,
            "contour_area": contour_area,
            "perimeter": perimeter,
            "circularity": circularity,
            "radial_std": radial_std,
            "border_irregularity": irregularity,
            "border_octant_score": b_oct,
            "symmetry": symmetry,
        }

        if rgb is not None and lab is not None and hsv is not None:
            row["mean_L"], row["mean_a"], row["mean_b"] = _mean_color(lab, component_mask)
            row["mean_H"], row["mean_S"], row["mean_V"] = _mean_color(hsv, component_mask)

            # --- Stage 13: 2-axis asymmetry (ABCD-A) ---
            h_asym, v_asym = asymmetry_score_2axis(component_mask)
            row["asymmetry_h"] = h_asym
            row["asymmetry_v"] = v_asym
            row["asymmetry_2axis_score"] = h_asym + v_asym

            # --- Stage 12: diameter (ABCD-D) ---
            d_eq, d_major = equivalent_diameter(component_mask)
            row["diameter_eq_px"] = d_eq
            row["diameter_major_px"] = d_major
            row["diameter_large_flag"] = bool(d_eq >= large_diameter_px)

            # --- Stage 11: named-color palette (ABCD-C) ---
            mask_pixels = lab[component_mask > 0].astype(np.float32)
            color_fracs = classify_dermo_colors(mask_pixels)
            for name in DERMO_COLOR_NAMES:
                row[f"pct_{name}"] = color_fracs[name]
            row["n_colors_present"] = color_fracs["n_colors_present"]

            # --- Stage 14: TDS ---
            A = row["asymmetry_2axis_score"]
            C = float(row["n_colors_present"])
            # Stage 1 leaves "D" undefined in the visual sense — we use a
            # simple proxy of the color-region count, capped at 5. Real
            # structure detection (Tier 2) will replace this with a real
            # count of pigment network / dots / streaks / etc.
            D = min(5.0, max(1.0, C))
            tds, tds_class = total_dermoscopy_score(A, b_oct, C, D)
            row["tds_score"] = tds
            row["tds_class"] = tds_class

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("area", ascending=False).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a metrics DataFrame to CSV. Always writes the header even if df is empty."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def df_to_jsonable(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records (NaN -> None, numpy -> python)."""
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")