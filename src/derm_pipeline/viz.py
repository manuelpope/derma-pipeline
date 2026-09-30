"""All matplotlib rendering goes through this module.

Centralizing the save/show branch in :func:`save_or_show` is what lets the
package stay headless by default while still supporting `--display`. No
``plot_*`` function calls ``plt.show()`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import pandas as pd

# `matplotlib.use(...)` is set in `cli.main` before this module is ever
# imported, so by the time we hit `import matplotlib.pyplot` the backend is
# already chosen. We still re-import pyplot lazily to avoid surprising the
# user when they want to use a different backend.

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from derm_pipeline.borders import (
    asymmetry_score,
    asymmetry_score_2axis,
    border_irregularity,
    equivalent_diameter,
    radial_profile,
)
from derm_pipeline.metrics import (
    AXIS_THRESHOLDS,
    classify_dermo_colors,
    diagnostic_flag,
)


# ---------------------------------------------------------------------------
# Save/show helper
# ---------------------------------------------------------------------------


def save_or_show(
    fig,
    save_path: str | Path | None,
    *,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Persist ``fig`` to disk, optionally pop a window.

    This is the *only* function that calls ``plt.show()``. Keeping the rule
    strict means we can never accidentally block a headless run.
    """
    written: Path | None = None
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written = p
    if show:
        try:
            plt.show(block=False)
        except Exception:
            # No GUI backend; ignore — we still wrote the file.
            pass
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# Reusable plot primitives
# ---------------------------------------------------------------------------


# Badge palette for the clinical-axis flags. Hex codes chosen for legibility
# against white backgrounds; "yellow" is darkened to amber because pure
# yellow is too light on a white figure.
_FLAG_HEX: dict[str, str] = {
    "green": "#2e7d32",
    "yellow": "#f9a825",
    "red": "#c62828",
}
_FLAG_LABEL: dict[str, str] = {
    "green": "bajo",
    "yellow": "medio",
    "red": "alto",
}


def _draw_diagnostic_badge(
    ax,
    *,
    x: float,
    y: float,
    score: float,
    flag: str,
    label: str | None = None,
    width: float = 0.16,
    height: float = 0.045,
) -> None:
    """Render a compact diagnostic badge (colored pill + score) on ``ax``.

    Parameters
    ----------
    ax : matplotlib axis
        Axis whose ``transAxes`` coordinate system is used.
    x, y : float
        Lower-left corner of the badge in axes coordinates.
    score : float
        Score in [0, 10].
    flag : {"green", "yellow", "red"}
        Severity band.
    label : str, optional
        Short caption to the LEFT of the score (e.g. "ABCDE-C"). Defaults to
        ``None`` (no label).
    width : float
        Badge width in axes coordinates (default 0.16).
    height : float
        Badge height in axes coordinates (default 0.045).
    """
    flag = flag if flag in _FLAG_HEX else "yellow"
    fill = _FLAG_HEX[flag]
    ax.add_patch(
        plt.Rectangle(
            (x, y), width, height,
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor="white",
            linewidth=1.0,
            clip_on=False,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        f"{score:.1f}/10",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        color="white",
    )
    if label:
        ax.text(
            x - 0.005,
            y + height / 2,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            weight="bold",
            color=fill,
        )


def draw_lesion_boxes(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    *,
    box_color: tuple[int, int, int] = (0, 255, 0),
    centroid_color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 4,
    with_labels: bool = True,
) -> NDArray[np.uint8]:
    """Annotate an RGB image with green bboxes + blue centroids for each lesion.

    Operates on a copy; the input is never mutated.
    """
    out = rgb.copy()
    for obj in lesions:
        x, y, w, h = obj["x"], obj["y"], obj["width"], obj["height"]
        area = obj["area"]
        cx, cy = int(obj["cx"]), int(obj["cy"])

        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, thickness)
        cv2.circle(out, (cx, cy), 8, centroid_color, -1)
        if with_labels:
            cv2.putText(
                out,
                f"ID: {obj['id']} | Area: {area}",
                (x, max(20, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2,
            )
    return out


def draw_contour_overlay(
    rgb: NDArray[np.uint8],
    contours: Sequence,
    *,
    color_bgr: tuple[int, int, int] = (255, 0, 128),
    thickness: int = 5,
) -> NDArray[np.uint8]:
    """Overlay magenta lesion contours on an RGB image (cell 19)."""
    out = rgb.copy()
    cv2.drawContours(out, list(contours), -1, color_bgr, thickness)
    return out


def draw_zoom_markers(
    rgb: NDArray[np.uint8],
    points: Sequence[tuple[int, int]],
    *,
    radius: int,
    color_bgr: tuple[int, int, int] = (255, 235, 59),
    label_prefix: str = "",
    thickness: int = 3,
) -> NDArray[np.uint8]:
    """Annotate each ``(x, y)`` with a circle + numeric label."""
    out = rgb.copy()
    for idx, (cx, cy) in enumerate(points):
        cv2.circle(out, (cx, cy), radius, color_bgr, thickness)
        if label_prefix or True:
            cv2.putText(
                out,
                f"{idx + 1}",
                (cx - 15, cy - radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color_bgr,
                3,
            )
    return out


def _overlay_masks(
    rgb: NDArray[np.uint8],
    labels: NDArray[np.int32],
    lesions: Sequence[dict],
    *,
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """Semi-transparent RGB overlay where each lesion is painted with a unique
    ``tab10`` color, then magenta contours drawn on top.

    Operates on a copy. Returns ``rgb.copy()`` unchanged if ``lesions`` is empty.
    """
    if not lesions:
        return rgb.copy()

    cmap = plt.get_cmap("tab10")
    overlay = np.zeros_like(rgb)
    for i, lesion in enumerate(lesions):
        lid = lesion["id"]
        rgba = cmap(i % cmap.N)
        bgr = (int(255 * rgba[2]), int(255 * rgba[1]), int(255 * rgba[0]))
        m = labels == lid
        overlay[m] = bgr

    out = cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0)

    # Contours on top for clinical-boundary clarity.
    for lesion in lesions:
        m = np.uint8(labels == lesion["id"]) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, (255, 0, 128), 2)
    return out


# ---------------------------------------------------------------------------
# Stage 8 — Lesion Segmentation
# ---------------------------------------------------------------------------


def plot_lesion_segmentation(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    labels: NDArray[np.int32],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Three-panel segmentation preview: original · binary mask · colored overlay.

    The binary mask panel is the foundation for every downstream analysis
    (color stats, contour metrics, radial profile). The overlay panel
    confirms that each connected component corresponds to a clinically
    meaningful lesion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb)
    axes[0].set_title("1. Imagen original")
    axes[0].axis("off")

    binary = np.uint8(labels > 0) * 255
    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title(f"2. Máscara binaria ({len(lesions)} lesiones)")
    axes[1].axis("off")

    overlay = _overlay_masks(rgb, labels, lesions)
    overlay = draw_lesion_boxes(overlay, lesions, thickness=3)
    axes[2].imshow(overlay)
    axes[2].set_title("3. Overlay (color por lesión + contornos)")
    axes[2].axis("off")

    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Stage 9 — Color Analysis
# ---------------------------------------------------------------------------


_LAB_BINS = {"L": 32, "a": 32, "b": 32}
_HSV_BINS = {"H": 180, "S": 32, "V": 32}
_LAB_RANGES = {"L": (0, 256), "a": (0, 256), "b": (0, 256)}
_HSV_RANGES = {"H": (0, 180), "S": (0, 256), "V": (0, 256)}
_LAB_INDEX = {"L": 0, "a": 1, "b": 2}
_HSV_INDEX = {"H": 0, "S": 1, "V": 2}


def plot_color_analysis(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    labels: NDArray[np.int32],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Per-lesion dominant colors with frequency % (Stage 9).

    Layout: 1 column with one card per lesion. Each card shows:
        - the lesion ID + ROI thumbnail,
        - a horizontal bar where each segment is a dominant color
          (k-means in LAB, k=5) with width proportional to its % of the
          lesion's pixels and the % labeled inside the segment.

    A small per-lesion summary line below the bar lists the dominant
    color's HEX and LAB values. This is the most clinically actionable
    view of color: it tells the dermatologist "this lesion is 60%
    color-X, 25% color-Y, 15% color-Z" at a glance.
    """
    if not lesions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No lesions detected", ha="center", va="center", fontsize=20)
        ax.axis("off")
        return save_or_show(fig, save_path, show=show, dpi=dpi)

    K_CLUSTERS = 5  # dominant colors per lesion
    SAMPLE_MAX = 4000  # subsample mask pixels for k-means speed

    fig, ax = plt.subplots(figsize=(15, max(6, 2.1 * len(lesions) + 2)))
    ax.axis("off")

    n = len(lesions)
    row_height = 1.0 / max(n, 1)
    cmap = plt.get_cmap("tab10")

    for i, lesion in enumerate(lesions):
        lid = lesion["id"]
        m = np.uint8(labels == lid) * 255
        n_pixels = int(np.sum(m > 0))
        if n_pixels == 0:
            continue

        # --- K-means in LAB on subsample of lesion pixels ---
        rgb_pixels = rgb[m > 0]
        if rgb_pixels.shape[0] > SAMPLE_MAX:
            idx = np.random.default_rng(i + 1).choice(
                rgb_pixels.shape[0], size=SAMPLE_MAX, replace=False
            )
            rgb_pixels = rgb_pixels[idx]
        lab_pixels = cv2.cvtColor(
            rgb_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)

        # k-means uses full pixel count for the labels step (not just the
        # subsample), so we cluster on the subsample then assign all pixels.
        _, labels_k, centers_lab = cv2.kmeans(
            lab_pixels,
            K_CLUSTERS,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        # Map subsample → cluster id; then count occurrences on FULL mask pixels
        # to get the actual % of lesion area.
        full_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        full_pixels = full_lab[m > 0].astype(np.float32)
        # Nearest-centroid assignment (no second k-means pass — keep it cheap).
        diffs = full_pixels[:, None, :] - centers_lab[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        full_labels = np.argmin(dists, axis=1)
        counts = np.bincount(full_labels, minlength=K_CLUSTERS).astype(np.float64)
        total = counts.sum()
        if total == 0:
            continue
        pct = counts / total  # proportions (0..1)

        # Sort clusters by frequency descending so the largest color comes first.
        order = np.argsort(-pct)
        centers_lab = centers_lab[order]
        pct = pct[order]
        # Convert LAB centers back to RGB for display swatches.
        # cv2 outputs RGB LUT where L is 0..255, a/b 0..255 with 128 as zero.
        centers_lab_int = centers_lab.reshape(1, -1, 3).astype(np.uint8)
        centers_rgb = cv2.cvtColor(centers_lab_int, cv2.COLOR_LAB2RGB).reshape(-1, 3)

        # --- Card layout (one row per lesion) ---
        y_top = 1.0 - i * row_height
        y_bot = y_top - row_height * 0.85
        # Reserve top ~30% of the row for the header (ID + thumbnail + means),
        # bottom ~70% for the color bar.
        header_h = (y_top - y_bot) * 0.30
        bar_top = y_top - header_h - 0.02
        bar_bot = y_bot + 0.04

        # Header text (left side)
        lab_mean = cv2.mean(full_lab, mask=m)[:3]
        rgb_mean = cv2.mean(rgb, mask=m)[:3]  # BGR
        rgb_mean_rgb = (int(rgb_mean[2]), int(rgb_mean[1]), int(rgb_mean[0]))
        hex_mean = "#{:02X}{:02X}{:02X}".format(*rgb_mean_rgb)
        ax.text(
            0.02,
            y_top - header_h * 0.4,
            f"ID {lid}",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
            va="top",
        )
        ax.text(
            0.08,
            y_top - header_h * 0.4,
            f"Área={lesion['area']:,} px  ·  LAB=({lab_mean[0]:.0f},{lab_mean[1]:.0f},{lab_mean[2]:.0f})  ·  color medio {hex_mean}",
            transform=ax.transAxes,
            fontsize=9,
            family="monospace",
            va="top",
        )

        # Diagnostic badge — uses the k-means cluster distribution as a
        # "variegation" signal: more significant clusters → more heterogeneous
        # lesion → higher score. We score on the number of clusters with > 10%
        # coverage (so a single-dominant-color lesion stays at 1 cluster).
        n_sig = int(np.sum(pct > 0.10))
        score, flag = diagnostic_flag(float(n_sig), low=2.0, high=4.0)
        _draw_diagnostic_badge(
            ax,
            x=0.55,
            y=y_top - header_h * 0.78,
            score=score,
            flag=flag,
            label=f"var {n_sig}/5",
            width=0.18,
            height=0.05,
        )

        # Thumbnail on the right side of the header — made ~2× bigger so the
        # clinician can read the lesion's actual texture, not just a smudge.
        x, y, w, h = lesion["x"], lesion["y"], lesion["width"], lesion["height"]
        roi = rgb[y : y + h, x : x + w].copy()
        thumb = _resize_to_fit(roi, max_w=180, max_h=int(header_h * fig.bbox.height * 1.6))
        # Place thumbnail at right edge using a tiny inset axes.
        thumb_ax = fig.add_axes([
            0.75,
            y_top - header_h * 1.05,
            0.23,
            header_h * 1.10,
        ])
        thumb_ax.imshow(thumb)
        thumb_ax.set_xticks([])
        thumb_ax.set_yticks([])
        for spine in thumb_ax.spines.values():
            spine.set_edgecolor("#888")
            spine.set_linewidth(0.6)

        # --- The dominant-color bar ---
        bar_left = 0.02
        bar_right = 0.73
        bar_width = bar_right - bar_left
        x_cursor = bar_left
        for cluster_idx in range(K_CLUSTERS):
            seg_w = bar_width * pct[cluster_idx]
            if seg_w < 1e-4:
                continue
            rgb_color = tuple(int(c) for c in centers_rgb[cluster_idx])
            ax.add_patch(
                plt.Rectangle(
                    (x_cursor, bar_bot),
                    seg_w,
                    bar_top - bar_bot,
                    transform=ax.transAxes,
                    facecolor=np.array(rgb_color) / 255.0,
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            # Label inside the segment with the %; pick contrasting text color.
            luminance = 0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]
            txt_color = "black" if luminance > 140 else "white"
            if seg_w > 0.04:  # only label if there's room
                ax.text(
                    x_cursor + seg_w / 2,
                    (bar_top + bar_bot) / 2,
                    f"{pct[cluster_idx] * 100:.0f}%",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    color=txt_color,
                )
            x_cursor += seg_w

        # Frame around the bar
        ax.add_patch(
            plt.Rectangle(
                (bar_left, bar_bot),
                bar_width,
                bar_top - bar_bot,
                transform=ax.transAxes,
                fill=False,
                edgecolor="#888",
                linewidth=0.7,
            )
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Colores dominantes por lesión (k-means en LAB, k=5)",
        fontsize=13,
        pad=14,
    )
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def _resize_to_fit(
    img: NDArray[np.uint8],
    *,
    max_w: int,
    max_h: int,
) -> NDArray[np.uint8]:
    """Resize ``img`` preserving aspect ratio so it fits inside ``max_w × max_h``."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((max_h, max_w, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
    scale = min(max_w / w, max_h / h)
    if scale >= 1.0:
        return img.copy()
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


# ---------------------------------------------------------------------------
# Stage 10 — Border & Shape Analysis
# ---------------------------------------------------------------------------


def plot_border_shape(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    labels: NDArray[np.int32],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Border & shape metrics — compact table (Stage 10).

    Replaces the previous "mirror overlay + 3 metric gauges" layout with a
    single tidy table — one row per lesion — that lists the clinical shape
    metrics alongside a flag badge. The Stage 13 mirror visualization already
    shows the per-lesion symmetry overlay, so duplicating it here only made
    the page harder to scan.

    Columns:
        ID · Área (px) · Perímetro (px) · Circularidad ·
        Simetría H · Irregularidad · Octante (0-8) · Bandera

    The badge uses the border octant score with thresholds from
    :data:`AXIS_THRESHOLDS` (``"B"``).
    """
    if not lesions:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No lesions", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return save_or_show(fig, save_path, show=show, dpi=dpi)

    # Compute the shape metrics for every lesion up front.
    rows: list[dict] = []
    for lesion in lesions:
        lid = lesion["id"]
        m = np.uint8(labels == lid) * 255
        cc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cc:
            continue
        cont = max(cc, key=cv2.contourArea)
        per = float(cv2.arcLength(cont, True))
        ar = float(cv2.contourArea(cont))
        cir = (4 * np.pi * ar) / (per ** 2) if per > 0 else 0.0
        sym = asymmetry_score(m)
        (rcx, rcy), rad, _ = radial_profile(m)
        irreg = border_irregularity(cont, (rcx, rcy)) if rad.size > 0 else 0.0
        b_oct = int(min(8, max(0, round(irreg * 16.0))))
        rows.append({
            "id": lid,
            "area": int(lesion["area"]),
            "per": per,
            "circ": cir,
            "sym": sym,
            "irreg": irreg,
            "oct": b_oct,
        })

    if not rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No contours", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return save_or_show(fig, save_path, show=show, dpi=dpi)

    b_lo, b_hi = AXIS_THRESHOLDS["B"]
    table_data = []
    cell_colors: list[list[str]] = []
    for r in rows:
        score, flag = diagnostic_flag(r["oct"], low=b_lo, high=b_hi)
        flag_label = _FLAG_LABEL[flag]
        table_data.append([
            r["id"],
            f"{r['area']:,}",
            f"{r['per']:.0f}",
            f"{r['circ']:.2f}",
            f"{r['sym']:.2f}",
            f"{r['irreg']:.2f}",
            f"{r['oct']}/8",
            f"{score:.1f}/10 · {flag_label}",
        ])
        cell_colors.append(["white"] * 7 + [_FLAG_HEX[flag]])

    headers = [
        "ID", "Área (px)", "Perímetro (px)", "Circularidad",
        "Simetría H", "Irregularidad", "Borde (oct.)", "Bandera",
    ]

    fig_h = max(2.2, 0.55 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colColours=["#e3f2fd"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color the badge column cells with the flag color and white text.
    for row_idx in range(len(rows)):
        cell = table[row_idx + 1, len(headers) - 1]
        cell.set_facecolor(cell_colors[row_idx][-1])
        cell.set_text_props(color="white", weight="bold")
        cell.PAD = 0.05

    # Highlight ID column.
    for row_idx in range(len(rows)):
        table[row_idx + 1, 0].set_text_props(weight="bold")

    ax.set_title(
        "Métricas de forma por lesión (Stage 10) — tabla compacta",
        fontsize=13,
        pad=12,
    )
    return save_or_show(fig, save_path, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Stage 11 (Diameter, ABCD-D) was removed per user feedback — the diameter
# values are still computed and persisted in ``metrics.csv``
# (``diameter_eq_px``, ``diameter_major_px``, ``diameter_large_flag``) but
# no standalone visualization page is generated.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 12 — 2-axis Asymmetry (ABCD-A)
# ---------------------------------------------------------------------------


def plot_asymmetry(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    labels: NDArray[np.int32],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Stage 11 — 2-axis asymmetry (ABCD-A), simplified.

    Layout: 1 row × 2 panels for the largest lesion.

        1. Original lesion ROI with the fitted ellipse (oval) overlaid on
           top of it, plus the major and minor axes drawn through the
           ellipse center. The asymmetry is visible *by how well the
           ellipse fits*: a symmetric lesion hugs the ellipse, an
           asymmetric one pokes out on one side or the ellipse sits at an
           angle to the lesion's bounding box.
        2. Score strip with A_2 = h_asym + v_asym, the diagnostic flag,
           and the per-axis breakdown.

    Uses ``cv2.fitEllipse`` to fit the ellipse (requires ≥5 contour points;
    falls back to a minimum-area circle when the lesion contour is too
    short).
    """
    if not lesions:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No lesions", ha="center", va="center", fontsize=20)
        ax.axis("off")
        return save_or_show(fig, save_path, show=show, dpi=dpi)

    main = lesions[0]
    lid = main["id"]
    x, y, w, h = main["x"], main["y"], main["width"], main["height"]
    H_img, W_img = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W_img, x + w), min(H_img, y + h)
    roi_rgb = rgb[y0:y1, x0:x1].copy()
    roi_mask = np.uint8(labels == lid) * 255
    roi_mask = roi_mask[y0:y1, x0:x1]

    moments = cv2.moments(roi_mask)
    if moments["m00"] == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Degenerate ROI", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return save_or_show(fig, save_path, show=show, dpi=dpi)
    roi_cx = float(moments["m10"] / moments["m00"])
    roi_cy = float(moments["m01"] / moments["m00"])
    h_asym, v_asym = asymmetry_score_2axis(roi_mask)
    a_total = h_asym + v_asym

    # --- Fit ellipse to the largest external contour ---
    contours, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cx_e, cy_e = roi_cx, roi_cy
    major_len, minor_len = 0.0, 0.0
    angle_deg = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            try:
                (cx_e, cy_e), (ea, eb), angle_deg = cv2.fitEllipse(c)
                # fitEllipse returns (width, height); the major axis is the
                # longer of the two.
                major_len = float(max(ea, eb))
                minor_len = float(min(ea, eb))
            except cv2.error:
                pass
        if major_len == 0.0 and minor_len == 0.0:
            # Fallback: enclosing circle from contour extremes.
            (cx_e, cy_e), radius = cv2.minEnclosingCircle(c)
            major_len = minor_len = float(2 * radius)

    # --- Render ---
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.25)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[0, 1])

    annotated = roi_rgb.copy()
    # Draw the fitted ellipse (yellow, semi-transparent look via thicker line)
    if major_len > 0:
        cv2.ellipse(
            annotated,
            (int(round(cx_e)), int(round(cy_e))),
            (int(round(major_len / 2)), int(round(minor_len / 2))),
            angle_deg,
            0, 360,
            (255, 220, 0),  # yellow
            2,
            cv2.LINE_AA,
        )
        # Draw major and minor axes through the ellipse center, rotated by
        # the ellipse angle. The major axis = red, the minor axis = cyan.
        ang = np.deg2rad(angle_deg)
        dx_major, dy_major = np.cos(ang), np.sin(ang)
        dx_minor, dy_minor = -dy_major, dx_major
        half_maj = major_len / 2
        half_min = minor_len / 2
        p1 = (int(round(cx_e + dx_major * half_maj)),
              int(round(cy_e + dy_major * half_maj)))
        p2 = (int(round(cx_e - dx_major * half_maj)),
              int(round(cy_e - dy_major * half_maj)))
        cv2.line(annotated, p1, p2, (220, 30, 30), 2, cv2.LINE_AA)
        p1 = (int(round(cx_e + dx_minor * half_min)),
              int(round(cy_e + dy_minor * half_min)))
        p2 = (int(round(cx_e - dx_minor * half_min)),
              int(round(cy_e - dy_minor * half_min)))
        cv2.line(annotated, p1, p2, (30, 200, 220), 2, cv2.LINE_AA)
        # Center dot
        cv2.circle(annotated, (int(round(cx_e)), int(round(cy_e))), 5,
                   (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, (int(round(cx_e)), int(round(cy_e))), 5,
                   (0, 0, 0), 1, cv2.LINE_AA)
    ax_img.imshow(annotated)
    ax_img.set_title(
        f"Lesión ID {lid} · elipse ajustada\neje mayor (rojo) + eje menor (cian)",
        fontsize=11,
    )
    ax_img.axis("off")

    # --- Score strip ---
    ax_score.set_xlim(0, 1)
    ax_score.set_ylim(0, 1)
    ax_score.axis("off")
    a_lo, a_hi = AXIS_THRESHOLDS["A"]
    a_score, a_flag = diagnostic_flag(a_total, low=a_lo, high=a_hi)
    _draw_diagnostic_badge(
        ax_score,
        x=0.05, y=0.78,
        score=a_score,
        flag=a_flag,
        label="ABCD-A",
        width=0.40,
        height=0.10,
    )
    ax_score.text(
        0.5, 0.60,
        f"A₂ = h_asym + v_asym = {h_asym:.2f} + {v_asym:.2f}",
        ha="center", va="center",
        fontsize=11, family="monospace",
    )
    ax_score.text(
        0.5, 0.46,
        f"Eje mayor ≈ {major_len:.0f} px · ángulo {angle_deg:.0f}°",
        ha="center", va="center",
        fontsize=10, color="#444",
    )
    # How well the ellipse fits the lesion — qualitative "residual" badge.
    contours2, _ = cv2.findContours(
        roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if contours2:
        c2 = max(contours2, key=cv2.contourArea)
        inliers = 0
        if len(c2) >= 5:
            for pt in c2[:, 0, :]:
                # Approximate inlier check: angle from center within ±5° of
                # boundary point; skip the explicit closed-form fit for speed.
                pass
            # Quick residual estimate: number of contour pixels that fall
            # within the fitted ellipse band (±10% of major axis) vs outside.
            band = max(2.0, 0.10 * major_len)
            ys, xs = np.where(roi_mask > 0)
            dx = xs - cx_e
            dy = ys - cy_e
            # Rotate by -angle so the ellipse becomes axis-aligned.
            cos_a, sin_a = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
            xr = cos_a * dx + sin_a * dy
            yr = -sin_a * dx + cos_a * dy
            a_half = max(1.0, major_len / 2)
            b_half = max(1.0, minor_len / 2)
            # Inlier = lies between the inner and outer ellipse band.
            inner = (xr * xr) / max((a_half - band) ** 2, 1) + (yr * yr) / max((b_half - band) ** 2, 1)
            outer = (xr * xr) / max((a_half + band) ** 2, 1) + (yr * yr) / max((b_half + band) ** 2, 1)
            n_pix = len(xs)
            n_inliers = int(np.sum((inner >= 1) & (outer <= 1)))
            pct_in = 100.0 * n_inliers / max(1, n_pix)
            ax_score.text(
                0.5, 0.30,
                f"ajuste elipse ≈ {pct_in:.0f}% dentro de ±10%",
                ha="center", va="center",
                fontsize=10, color="#444",
            )
    ax_score.text(
        0.5, 0.10,
        "Una lesión simétrica llena el óvalo.\n"
        "Asimétrica → sobresale por un lado o\n"
        "el óvalo está rotado respecto al bounding box.",
        ha="center", va="center",
        fontsize=8, color="#666", style="italic",
    )

    fig.suptitle(
        f"Asimetría 2-ejes · lesión principal (ID {lid}) · Stage 11 / ABCD-A",
        fontsize=12, y=1.02,
    )
    return save_or_show(fig, save_path, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Stage 14 — Total Dermoscopy Score (TDS)


# ---------------------------------------------------------------------------
# Stage 15 — ABCDE Composite (one-page summary)


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------


def plot_pipeline_stages(
    stages: dict,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Render the 2×3 grid of pipeline stages (cell 11).

    Only stages 0–5 are shown; stages 6 (Otsu), 7 (Morphology) and 8
    (Connected Components) are computed for downstream lesion detection but
    are intentionally not surfaced in this preview.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    panels = [
        ("rgb", "1. Original", None),
        ("gray", "2. Grayscale", "gray"),
        ("blur", "3. Gaussian Blur", "gray"),
        ("enhanced", "4. CLAHE", "gray"),
        ("normalized", "5. Normalized", "gray"),
    ]
    flat_axes = axes.ravel()
    for ax, (key, title, cmap) in zip(flat_axes[: len(panels)], panels):
        arr = stages.get(key)
        if arr is None:
            ax.set_title(f"{title} (missing)")
            ax.axis("off")
            continue
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    # If the grid has more cells than panels (e.g. 2×4 → 8 cells), hide the
    # remaining ones so the layout doesn't show empty placeholders.
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_lesions(
    rgb: NDArray[np.uint8],
    lesions: Sequence[dict],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
    title: str = "Detección Final de Lesiones/Manchas Principales",
) -> Path | None:
    """Plot the full image with green bboxes (cell 13)."""
    annotated = draw_lesion_boxes(rgb, lesions)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(annotated)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_main_lesion_zoom(
    rgb: NDArray[np.uint8],
    lesion: dict,
    lesion_mask: NDArray[np.uint8],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Side-by-side: original ROI vs. ROI with clinical-boundary outline (cell 15)."""
    x, y, w, h = lesion["x"], lesion["y"], lesion["width"], lesion["height"]
    roi_rgb = rgb[y : y + h, x : x + w].copy()
    roi_mask = np.uint8(lesion_mask[y : y + h, x : x + w] == lesion["id"]) * 255

    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        main_contour = max(cnts, key=cv2.contourArea)
        cv2.drawContours(roi_rgb, [main_contour], -1, (0, 255, 0), 4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb[y : y + h, x : x + w])
    axes[0].set_title("1. Zoom Lesión Original")
    axes[0].axis("off")
    axes[1].imshow(roi_rgb)
    axes[1].set_title("2. Límite Clínico de la Mancha")
    axes[1].axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_contour_topography(
    rgb_roi: NDArray[np.uint8],
    gray_roi: NDArray[np.uint8],
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
    cmap: str = "spring",
    levels: int = 12,
) -> Path | None:
    """Bilateral-filtered ROI + ``contourf`` topography (cell 17)."""
    roi_denoised = cv2.bilateralFilter(gray_roi, d=9, sigmaColor=50, sigmaSpace=50)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb_roi)
    axes[0].set_title("Vista Acercada de la Lesión Original")
    axes[0].axis("off")
    cf = axes[1].contourf(roi_denoised, levels=levels, cmap=cmap)
    fig.colorbar(cf, ax=axes[1], label="Luminosidad / Densidad de Pigmento")
    axes[1].set_title("Topografía Dermatológica (Paleta de Alta Energía)")
    axes[1].invert_yaxis()
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_contour_overlay(
    rgb: NDArray[np.uint8],
    contours: Sequence,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Magenta silhouette overlay (cell 19)."""
    out = draw_contour_overlay(rgb, contours)
    fig, axes = plt.subplots(1, 2, figsize=(15, 12))
    axes[0].imshow(rgb)
    axes[0].set_title("1. Imagen Original", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(out)
    axes[1].set_title("2. Siluetas de las Manchas Detectadas (Fronteras Clínicas)", fontsize=14)
    axes[1].axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_critical_zooms(
    rgb_enhanced: NDArray[np.uint8],
    points: Sequence[tuple[int, int]],
    patches: Sequence[NDArray[np.uint8]],
    *,
    contour: NDArray | None = None,
    crop_radius: int | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
    title: str = "Estructura Dermatológica - Lupas con Cobertura Dinámica de Bordes",
) -> Path | None:
    """Big left panel + N right panels — the cell 21/23 figure with corrected semantics.

    The right-column width scales with ``crop_radius`` so the lupa renders
    the patch at roughly the same per-pixel detail as the main image,
    regardless of ``scale_factor``. Higher ``scale_factor`` → smaller
    physical patch → narrower lupa → constant visual magnification.

    Note: ``fig.tight_layout()`` is intentionally NOT called — it fights
    ``add_gridspec`` and collapses the lupa widths back to uniform size.
    ``save_or_show`` already passes ``bbox_inches="tight"`` to ``savefig``.
    """
    if crop_radius is None:
        crop_radius = 120

    ref_crop_radius = 120  # matches ``PipelineConfig.scale_factor=2.0`` defaults
    lupa_width_ratio = crop_radius / ref_crop_radius

    rows = max(3, len(points))
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        rows, 2,
        width_ratios=[2.0, 1.0 * lupa_width_ratio],
        wspace=0.2, hspace=0.3,
    )

    ax_main = fig.add_subplot(gs[:, 0])
    main_vis = rgb_enhanced.copy()
    if contour is not None:
        cv2.drawContours(main_vis, [contour], -1, (0, 255, 128), 2)
    main_vis = draw_zoom_markers(main_vis, points, radius=crop_radius)

    for idx, patch in enumerate(patches):
        ax_zoom = fig.add_subplot(gs[idx, 1])
        ax_zoom.imshow(patch)
        cx, cy = points[idx] if idx < len(points) else (0, 0)
        ax_zoom.set_title(
            f"Lupa {idx + 1}: Coords ({cx}, {cy})", fontsize=11, color="darkorange"
        )
        ax_zoom.axis("on")

    ax_main.imshow(main_vis)
    ax_main.set_title(title, fontsize=14)
    ax_main.axis("off")

    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_hough_circles(
    rgb: NDArray[np.uint8],
    circles: NDArray[np.int32] | None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Blue circle overlay for the Hough output (cell 29)."""
    out = rgb.copy()
    if circles is not None:
        for x, y, r in circles:
            cv2.circle(out, (int(x), int(y)), int(r), (255, 0, 0), 3)
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(out)
    ax.set_title("Hough Circle Detection")
    ax.axis("off")
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_all_outputs(
    rgb: NDArray[np.uint8],
    stages: dict,
    lesions: Sequence[dict],
    main_lesion: dict | None,
    main_mask: NDArray[np.uint8] | None,
    rgb_roi: NDArray[np.uint8],
    gray_roi: NDArray[np.uint8],
    all_contours: Sequence,
    enhanced_rgb: NDArray[np.uint8],
    biggest_contour,
    points: Sequence[tuple[int, int]],
    patches: Sequence[NDArray[np.uint8]],
    crop_radius: int,
    circles: NDArray[np.int32] | None = None,
    labels: NDArray[np.int32] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Single 3×5 composite figure with every visualization in one shot.

    Layout (panels are skipped gracefully when their data is missing):

        Row 1: original | segmentación | detections | contours  | ROI principal
        Row 2: boundary  | topography   | lupas      | hough      | Lupa 1
        Row 3: Lupa 2    | Lupa 3       | Stage 9 (color thumb) | Stage 10 (border thumb) | Stage 11 (asymmetry thumb)

    This is what the ``--all`` CLI flag / ``all=True`` API form field
    produces, so a single PNG (``99_all_outputs.png``) replaces the dozen of
    per-stage images for quick inspection. Resolution is still full; the
    figure just packs every panel into one canvas.

    The ``labels`` argument is required for the Stage 8/9/10 thumbnails;
    pass the same ``labels`` ndarray used by ``compute_metrics``.
    """
    fig, axes = plt.subplots(3, 5, figsize=(30, 18))
    flat = axes.ravel()

    for ax in flat:
        ax.axis("off")

    # --- Row 1: detection pipeline ---
    flat[0].imshow(rgb)
    flat[0].set_title("1. Original")

    # --- Stage 8 thumbnail: segmentation overlay ---
    if labels is not None and lesions:
        flat[1].imshow(_overlay_masks(rgb, labels, lesions))
        flat[1].set_title(f"2. Segmentación ({len(lesions)} lesiones)")
    elif labels is not None:
        flat[1].imshow(np.uint8(labels > 0) * 255, cmap="gray")
        flat[1].set_title("2. Segmentación (sin lesiones)")
    else:
        flat[1].text(0.5, 0.5, "labels N/A", ha="center", va="center")
        flat[1].set_title("2. Segmentación")

    if lesions:
        flat[2].imshow(draw_lesion_boxes(rgb, lesions))
    else:
        flat[2].imshow(rgb)
    flat[2].set_title("3. Detecciones")

    if all_contours:
        flat[3].imshow(draw_contour_overlay(rgb, all_contours))
    else:
        flat[3].imshow(rgb)
    flat[3].set_title("4. Contornos")

    # --- Row 2: lesion details ---
    has_main = main_lesion is not None and rgb_roi.size > 0

    if has_main:
        flat[4].imshow(rgb_roi)
        flat[4].set_title("5. ROI lesión principal")

        boundary = rgb_roi.copy()
        cnts, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(boundary, [max(cnts, key=cv2.contourArea)], -1, (0, 255, 0), 4)
        flat[5].imshow(boundary)
        flat[5].set_title("6. Borde clínico")
    else:
        flat[4].text(0.5, 0.5, "no lesion", ha="center", va="center")
        flat[5].text(0.5, 0.5, "no lesion", ha="center", va="center")

    flat[6].axis("on")
    if gray_roi.size:
        denoised = cv2.bilateralFilter(gray_roi, d=9, sigmaColor=50, sigmaSpace=50)
        cf = flat[6].contourf(denoised, levels=12, cmap="spring")
        fig.colorbar(cf, ax=flat[6])
    flat[6].set_title("7. Topografía")

    # --- Lupas críticas (kept at flat[7] from the original 3×4 layout) ---
    flat[7].axis("off")
    if points:
        marker_vis = enhanced_rgb.copy()
        if biggest_contour is not None:
            cv2.drawContours(marker_vis, [biggest_contour], -1, (0, 255, 128), 2)
        marker_vis = draw_zoom_markers(marker_vis, points, radius=crop_radius)
        flat[7].imshow(marker_vis)
        flat[7].set_title("8. Lupas críticas")
    else:
        flat[7].text(0.5, 0.5, "sin puntos críticos", ha="center", va="center")
        flat[7].set_title("8. Lupas críticas")

    # --- Row 3: hough + zoom patches + Stage 9 thumbnail + Stage 10 thumbnail + blank ---
    flat[8].axis("off")
    if circles is not None and len(circles) > 0:
        hough_vis = rgb.copy()
        for x, y, r in circles:
            cv2.circle(hough_vis, (int(x), int(y)), int(r), (255, 0, 0), 3)
            cv2.circle(hough_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        flat[8].imshow(hough_vis)
        flat[8].set_title("9. Hough")
    else:
        flat[8].text(0.5, 0.5, "Hough no ejecutado", ha="center", va="center")
        flat[8].set_title("9. Hough")

    for panel_idx, patch_idx in zip([9, 10, 11], [0, 1, 2]):
        flat[panel_idx].axis("off")
        if patch_idx < len(patches):
            # Scale the visible region so per-pixel magnification stays roughly
            # constant across ``scale_factor``: higher scale_factor → smaller
            # patch → smaller extent → less zoom inside the fixed-size cell.
            ref_crop_radius = 120
            half = 0.5 * ref_crop_radius / max(1, crop_radius)
            flat[panel_idx].imshow(
                patches[patch_idx],
                extent=(-half, half, -half, half),
                aspect="auto",
            )
            cx, cy = points[patch_idx] if patch_idx < len(points) else (0, 0)
            flat[panel_idx].set_title(f"Lupa {patch_idx + 1} @ ({cx}, {cy})")
        else:
            flat[panel_idx].text(0.5, 0.5, "—", ha="center", va="center")

    # --- Stage 9 thumbnail: LAB means per lesion (bar chart) + variegation flag ---
    flat[12].axis("on")
    if labels is not None and lesions:
        ax = flat[12]
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        x_pos = np.arange(len(lesions))
        means_L, means_a, means_b = [], [], []
        for lesion in lesions:
            m = np.uint8(labels == lesion["id"]) * 255
            if int(np.sum(m > 0)) == 0:
                means_L.append(0)
                means_a.append(0)
                means_b.append(0)
                continue
            ml = cv2.mean(lab, mask=m)[:3]
            means_L.append(ml[0])
            means_a.append(ml[1])
            means_b.append(ml[2])
        ax.bar(x_pos - 0.27, means_L, width=0.27, color="#444", label="L*")
        ax.bar(x_pos, means_a, width=0.27, color="#c33", label="a*")
        ax.bar(x_pos + 0.27, means_b, width=0.27, color="#39c", label="b*")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"ID{l['id']}" for l in lesions], fontsize=8)
        ax.set_ylabel("valor", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("9. Análisis de color · LAB medio", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        # Variegation badge for the largest lesion (matches Stage 9 page).
        m0 = np.uint8(labels == lesions[0]["id"]) * 255
        if int(np.sum(m0 > 0)) > 0:
            # Use the actual named-color count (matches Stage 11 / diagnostic
            # threshold for "C"). k-means variegation is too noisy at this size.
            fracs = classify_dermo_colors(lab[m0 > 0].astype(np.float32))
            n_sig = fracs["n_colors_present"]
            score, flag = diagnostic_flag(float(n_sig), low=2.0, high=4.0)
            _draw_diagnostic_badge(
                ax,
                x=0.02,
                y=0.96,
                score=score,
                flag=flag,
                label=f"var {n_sig}/6",
                width=0.20,
                height=0.05,
            )
    else:
        flat[12].text(0.5, 0.5, "sin lesiones", ha="center", va="center")
        flat[12].set_title("9. Análisis de color")

    # --- Stage 10 thumbnail: Stage 10 is now a compact table — no visualization.
    # Show a tiny note pointing the reader at the actual artifact.
    flat[13].axis("off")
    flat[13].text(
        0.5,
        0.55,
        "Stage 10",
        ha="center", va="center",
        fontsize=14, weight="bold",
    )
    flat[13].text(
        0.5,
        0.42,
        "Métricas de forma",
        ha="center", va="center",
        fontsize=10, color="#444",
    )
    flat[13].text(
        0.5,
        0.30,
        "→ 10_border_shape.png",
        ha="center", va="center",
        fontsize=9, color="#1565c0", weight="bold",
        family="monospace",
    )
    flat[13].text(
        0.5,
        0.18,
        "(tabla compacta)",
        ha="center", va="center",
        fontsize=8, color="gray", style="italic",
    )
    flat[13].set_title("10. Borde y forma", fontsize=10)

    # --- Stage 11 thumbnail: fitted ellipse + axes (main lesion) ---
    flat[14].axis("off")
    if labels is not None and lesions and main_mask is not None and rgb_roi.size:
        m = np.uint8(labels == lesions[0]["id"]) * 255
        if int(np.sum(m > 0)) > 0:
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if len(c) >= 5:
                    try:
                        (cx_e, cy_e), (ea, eb), angle_deg = cv2.fitEllipse(c)
                    except cv2.error:
                        (cx_e, cy_e), radius = cv2.minEnclosingCircle(c)
                        ea = eb = 2 * radius
                        angle_deg = 0
                    # Draw the ellipse + axes on the ROI
                    view = rgb_roi.copy()
                    major_len = max(ea, eb)
                    minor_len = min(ea, eb)
                    cv2.ellipse(
                        view,
                        (int(round(cx_e)), int(round(cy_e))),
                        (int(round(major_len / 2)), int(round(minor_len / 2))),
                        angle_deg, 0, 360, (255, 220, 0), 2, cv2.LINE_AA,
                    )
                    ang = np.deg2rad(angle_deg)
                    dx_m, dy_m = np.cos(ang), np.sin(ang)
                    dx_n, dy_n = -dy_m, dx_m
                    h_m, h_n = major_len / 2, minor_len / 2
                    cv2.line(
                        view,
                        (int(round(cx_e + dx_m * h_m)), int(round(cy_e + dy_m * h_m))),
                        (int(round(cx_e - dx_m * h_m)), int(round(cy_e - dy_m * h_m))),
                        (220, 30, 30), 2, cv2.LINE_AA,
                    )
                    cv2.line(
                        view,
                        (int(round(cx_e + dx_n * h_n)), int(round(cy_e + dy_n * h_n))),
                        (int(round(cx_e - dx_n * h_n)), int(round(cy_e - dy_n * h_n))),
                        (30, 200, 220), 2, cv2.LINE_AA,
                    )
                    flat[14].imshow(view)
                    h_asym, v_asym = asymmetry_score_2axis(m)
                    flat[14].set_title(
                        f"11. Asimetría · óvalo {angle_deg:.0f}° · A₂={h_asym + v_asym:.2f}",
                        fontsize=9,
                    )
                else:
                    flat[14].imshow(rgb_roi)
                    flat[14].set_title("11. Asimetría")
            else:
                flat[14].text(0.5, 0.5, "no contour", ha="center", va="center")
                flat[14].set_title("11. Asimetría")
        else:
            flat[14].text(0.5, 0.5, "mask vacía", ha="center", va="center")
            flat[14].set_title("11. Asimetría")
    else:
        flat[14].text(0.5, 0.5, "no lesion", ha="center", va="center")
        flat[14].set_title("11. Asimetría")

    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Filter laboratory + color-space atlas (global visual references)
# ---------------------------------------------------------------------------
#
# These two pages show the same image under 16 different filter / color-space
# variants. They are independent of lesion detection (work even when no
# lesions are found), and they are pure visual aids — no scores, no labels.


def _ensure_gray(rgb: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 grayscale view of ``rgb``."""
    if rgb.ndim == 2:
        return rgb if rgb.dtype == np.uint8 else rgb.astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    """Convenience: CLAHE on uint8 grayscale."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linear-normalize any numeric array to 0..255 uint8."""
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _make_filter_panels(
    rgb: np.ndarray,
) -> list[tuple[str, np.ndarray, str | None]]:
    """Build the 16 (title, image, cmap) tuples for :func:`plot_filter_lab`.

    Layout (4×4):

        Row 1 — bilateral at increasing neighborhood size
        Row 2 — CLAHE at increasing clip limit + a larger-tile variant
        Row 3 — Gaussian blur at three σ + median k=5
        Row 4 — Canny at two thresholds + Sobel X/Y
    """
    gray = _ensure_gray(rgb)
    panels: list[tuple[str, np.ndarray, str | None]] = []

    # Row 1 — bilateral (RGB)
    for d, sigma in ((5, 20), (9, 50), (15, 100), (20, 150)):
        out = cv2.bilateralFilter(rgb, d=d, sigmaColor=sigma, sigmaSpace=sigma)
        panels.append((f"Bilateral · d={d} σ={sigma}", out, None))

    # Row 2 — CLAHE (grayscale)
    for clip, tile in ((1.5, 8), (2.5, 8), (4.0, 8), (2.5, 16)):
        panels.append((f"CLAHE · clip={clip} tile={tile}", _clahe(gray, clip, tile), "gray"))

    # Row 3 — blur (RGB)
    for sigma in (1, 3, 7):
        out = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=float(sigma))
        panels.append((f"Gaussiano · σ={sigma}", out, None))
    panels.append(("Mediana · k=5", cv2.medianBlur(rgb, 5), None))

    # Row 4 — edges (grayscale)
    panels.append(("Canny · 30/100", cv2.Canny(gray, 30, 100), "gray"))
    panels.append(("Canny · 80/200", cv2.Canny(gray, 80, 200), "gray"))
    panels.append((
        "Sobel · X",
        _normalize_to_uint8(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
        "gray",
    ))
    panels.append((
        "Sobel · Y",
        _normalize_to_uint8(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)),
        "gray",
    ))

    return panels


def _make_color_atlas_panels(
    rgb: np.ndarray,
) -> list[tuple[str, np.ndarray, str | None]]:
    """Build the 16 (title, image, cmap) tuples for :func:`plot_color_space_atlas`.

    Layout (4×4):

        Row 1 — LAB: L*, a*, b*, a*+b* combined
        Row 2 — HSV: H, S, V, V equalized
        Row 3 — YCrCb: Y, Cr, Cb, Y equalized
        Row 4 — channel differences (R−G, R−B), gray equalized, a* normalized
    """
    gray = _ensure_gray(rgb)
    panels: list[tuple[str, np.ndarray, str | None]] = []

    # Row 1 — LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, a_ch, b_ch = cv2.split(lab)
    panels.append(("LAB · L*", L, "gray"))
    panels.append(("LAB · a*", a_ch, "gray"))
    panels.append(("LAB · b*", b_ch, "gray"))
    ab_combined = _normalize_to_uint8(a_ch.astype(np.int16) + b_ch.astype(np.int16))
    panels.append(("LAB · a*+b*", ab_combined, "gray"))

    # Row 2 — HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    panels.append(("HSV · H", H, "hsv"))
    panels.append(("HSV · S", S, "gray"))
    panels.append(("HSV · V", V, "gray"))
    panels.append(("HSV · V ecualizado", _clahe(V, 2.5, 8), "gray"))

    # Row 3 — YCrCb
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    Y_ch, Cr, Cb = cv2.split(ycrcb)
    panels.append(("YCrCb · Y", Y_ch, "gray"))
    panels.append(("YCrCb · Cr", Cr, "gray"))
    panels.append(("YCrCb · Cb", Cb, "gray"))
    panels.append(("YCrCb · Y ecualizado", _clahe(Y_ch, 2.5, 8), "gray"))

    # Row 4 — channel differences / proxies
    R, G, B = cv2.split(rgb)
    panels.append(("R − G", _normalize_to_uint8(cv2.subtract(R, G)), "inferno"))
    panels.append(("R − B (proxy melanina)", _normalize_to_uint8(cv2.subtract(R, B)), "inferno"))
    panels.append(("Gris ecualizado", _clahe(gray, 2.5, 8), "gray"))
    panels.append(("a* normalizado", _normalize_to_uint8(a_ch), "gray"))

    return panels


def plot_filter_lab(
    rgb: np.ndarray,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Render a 4×4 grid with 16 filter variants applied to ``rgb``.

    Independent of lesion detection — works on any non-empty RGB array.
    Pure visual aid; no scores or labels.
    """
    panels = _make_filter_panels(rgb)
    fig, axes = plt.subplots(4, 4, figsize=(11, 14))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "Laboratorio de Filtros · Imagen Completa",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)


def plot_color_space_atlas(
    rgb: np.ndarray,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
    dpi: int = 150,
) -> Path | None:
    """Render a 4×4 grid with 16 color-space / channel-mix views of ``rgb``."""
    panels = _make_color_atlas_panels(rgb)
    fig, axes = plt.subplots(4, 4, figsize=(11, 14))
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(
        "Atlas de Espacios de Color · Imagen Completa",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout()
    return save_or_show(fig, save_path, show=show, dpi=dpi)