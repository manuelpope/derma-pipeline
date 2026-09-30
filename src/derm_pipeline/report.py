"""Multi-page PDF report assembly.

Picks up the PNGs + JSON summary that ``cli.run()`` already wrote, and glues
them into a single A4 PDF suitable for sharing with a clinician or archiving
in a patient record. Uses matplotlib's own ``PdfPages`` so we don't add a new
dependency.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


# Each entry is (filename, human-readable title, matplotlib cmap).
# Pass `cmap="gray"` for single-channel PNGs so matplotlib doesn't fall back
# to the default `viridis` ramp (which paints grayscale images in
# green/yellow/purple instead of true grayscale).
STAGE_PAGES: tuple[tuple[str, str, str | None], ...] = (
    ("00_original.png", "Stage 0 · Original RGB", None),
    ("01_gray.png", "Stage 1 · Grayscale", "gray"),
    ("02_bilateral.png", "Stage 2 · Bilateral Filter (edge-preserving smoothing)", "gray"),
    ("03_blur.png", "Stage 3 · Gaussian Blur", "gray"),
    ("04_enhanced.png", "Stage 4 · CLAHE — local contrast enhancement", "gray"),
    ("05_normalized.png", "Stage 5 · Min/Max Normalization", "gray"),
)
OUTPUT_PAGES: tuple[tuple[str, str, str | None], ...] = (
    ("12_detected.png", "Lesion Detection (bboxes + IDs)", None),
    ("13_main_lesion_zoom.png", "Main Lesion · Before / Boundary", None),
    ("14_topography.png", "Topography (contourf)", None),
    ("15_contours_only.png", "Lesion Contour Silhouettes", None),
    ("16_critical_zooms.png", "Critical-Border Dynamic Zooms", None),
)
# Stage 8/9/10 analysis pages — emitted after the preprocessing stages and
# before the legacy post-detection outputs. They are independent of Hough
# and of lesion-dependent detection (skipped silently if the PNG is absent).
ANALYSIS_PAGES: tuple[tuple[str, str, str | None], ...] = (
    ("08_segmentation.png", "Stage 8 · Lesion Segmentation", None),
    ("09_color_analysis.png", "Stage 9 · Color Analysis (k-means LAB)", None),
    ("10_border_shape.png", "Stage 10 · Border & Shape Analysis", None),
)
# Stage 11 — Tier 1 ABCDE additions. Always on, rendered between the
# legacy ANALYSIS_PAGES and the post-detection OUTPUT_PAGES. Stage 11
# (Diameter, ABCD-D) was removed per user feedback; the diameter values
# remain in metrics.csv.
ABCDE_PAGES: tuple[tuple[str, str, str | None], ...] = (
    ("11_asymmetry.png", "Stage 11 · 2-Axis Asymmetry (ABCD-A)", None),
)
ALL_PAGE: tuple[str, str, str | None] = ("99_all_outputs.png", "Composite Overview (3×5)", None)
HOUGH_PAGE: tuple[str, str, str | None] = ("17_hough_circles.png", "Hough Circle Detection", None)
GRID_PAGE: tuple[str, str, str | None] = (
    "00_pipeline_stages.png",
    "Pipeline Stages Summary (2×3 grid)",
    None,
)
# Two global visual-reference pages appended after OUTPUT_PAGES and before Hough.
# Independent of lesion detection — they always render as long as the PNG exists.
FILTER_LAB_PAGE: tuple[str, str, str | None] = (
    "18_filter_lab.png",
    "Laboratorio de Filtros · Imagen Completa",
    None,
)
COLOR_ATLAS_PAGE: tuple[str, str, str | None] = (
    "19_color_atlas.png",
    "Atlas de Espacios de Color · Imagen Completa",
    None,
)


def _embed_image_page(
    pdf: PdfPages,
    png_path: Path,
    *,
    title: str,
    cmap: str | None = None,
    max_inches: float = 10.0,
) -> None:
    """Append ``png_path`` to ``pdf`` as a new page, sized to its aspect ratio.

    ``cmap`` is forwarded to ``ax.imshow`` so single-channel PNGs render with
    the chosen colormap (default ``"gray"`` for 2D arrays to avoid the
    misleading viridis ramp).
    """
    img = mpimg.imread(png_path)
    # Defensive: if a 2D array sneaks in without an explicit cmap, force
    # grayscale — the matplotlib default is viridis which is a *bad* choice
    # for clinical grayscale imagery.
    if cmap is None and img.ndim == 2:
        cmap = "gray"

    h, w = img.shape[:2]
    aspect = w / max(h, 1)
    if aspect >= 1.0:
        fig_w = max_inches
        fig_h = max(3.5, max_inches / aspect)
    else:
        fig_h = max_inches
        fig_w = max(3.5, max_inches * aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.imshow(img, cmap=cmap)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    pdf.savefig(fig)
    plt.close(fig)


def _cover_page(pdf: PdfPages, summary: dict, *, hough: bool) -> None:
    """Title page with image metadata, parameters, and result counts."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.axis("off")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows: list[tuple[str, str, str]] = [
        ("title", "Dermatoscopio Pipeline Report", ""),
        ("blank", "", ""),
        ("subtitle", f"Generated: {ts}", ""),
        ("blank", "", ""),
        ("section", "Input", ""),
        ("text", "Filename:", str(Path(str(summary.get("input", "?"))).name)),
        ("text", "Shape (HxWxC):", str(tuple(summary.get("image_shape", [])))),
        ("blank", "", ""),
        ("section", "Configuration", ""),
        ("text", "scale_factor:", str(summary.get("scale_factor"))),
        ("text", "num_zooms:", str(summary.get("num_zooms_requested"))),
        ("text", "min_area:", str(summary.get("min_area"))),
        ("text", "min_dist:", str(summary.get("min_dist", "—"))),
        ("text", "zoom_base_radius:", str(summary.get("zoom_base_radius"))),
        ("text", "zoom_relative:", str(summary.get("zoom_relative", False))),
        ("text", "hough:", "enabled" if hough else "disabled"),
        ("blank", "", ""),
        ("section", "Results", ""),
        ("text", "Lesions detected:", str(summary.get("num_lesions", 0))),
        ("text", "Zoom points placed:", str(summary.get("num_zooms_placed", "—"))),
    ]
    if "crop_radius_pixels" in summary:
        rows.append(("text", "Crop radius:", f"{summary['crop_radius_pixels']} px"))
    if "patch_size" in summary:
        rows.append(("text", "Patch size:", f"{tuple(summary['patch_size'])}"))
    if "zoom_base_radius_effective" in summary:
        rows.append(
            ("text", "Effective zoom radius:", f"{summary['zoom_base_radius_effective']} px")
        )

    warnings = summary.get("warnings") or []
    if warnings:
        rows.append(("blank", "", ""))
        rows.append(("section", "Warnings", ""))
        for w in warnings:
            rows.append(("text", "•", str(w)))

    # styles
    styles = {
        "title": dict(size=22, weight="bold", color="#1a1a1a"),
        "subtitle": dict(size=10, color="#555"),
        "section": dict(size=13, weight="bold", color="#0d47a1"),
        "text": dict(size=10, color="#222"),
        "blank": dict(size=6, color="#fff"),
    }

    y = 0.95
    line_height = 0.032
    for kind, label, value in rows:
        st = styles[kind]
        if kind == "blank":
            y -= line_height
            continue
        if kind == "section":
            y -= line_height * 0.6
            ax.text(0.05, y, label, transform=ax.transAxes, **st)
            y -= line_height * 1.2
            continue
        if kind == "title":
            ax.text(0.05, y, label, transform=ax.transAxes, **st)
            y -= line_height * 2.0
            continue
        ax.text(0.05, y, label, transform=ax.transAxes, **st)
        if value:
            ax.text(0.45, y, value, transform=ax.transAxes, fontsize=st["size"])
        y -= line_height

    pdf.savefig(fig)
    plt.close(fig)


def build_report_pdf(
    output_dir: str | Path,
    summary: dict,
    *,
    hough: bool = False,
    all_outputs: bool = False,
    expand_stages: bool = True,
    filter_lab: bool = True,
    color_atlas: bool = True,
    filename: str | None = None,
) -> Path:
    """Assemble ``report.pdf`` inside ``output_dir`` and return the path.

    Page order (matches the original Colab cell-11 style as a grid summary,
    then drill down per-stage, then measurement, then post-detection outputs):

        1. Cover                       (parameters + counts + warnings)
        2. Pipeline Stages 2x3 grid    (the original notebook cell-11 figure;
                                       faithful to ``img_filtros_dermatoscopio``)
        3-N. Each preprocessing stage as its own page (when expand_stages=True):
             00_original, 01_gray, 02_bilateral, 03_blur, 04_enhanced,
             05_normalized
        N+1.  Stage 8 · Lesion Segmentation       (08_segmentation.png)
        N+2.  Stage 9 · Color Analysis            (09_color_analysis.png)
        N+3.  Stage 10 · Border & Shape           (10_border_shape.png)
        N+4.  Stage 11 · 2-axis Asymmetry (A)     (11_asymmetry.png)
        Then: 99_all_outputs.png       (3x5 composite, only if all_outputs=True)
        Then: 13_detected, 14_main_lesion_zoom, 15_topography,
              16_contours_only, 17_critical_zooms
        Then: 19_filter_lab.png       (only if filter_lab=True; global page)
              20_color_atlas.png      (only if color_atlas=True; global page)
        Last: 18_hough_circles.png     (only if hough ran)

    Set ``expand_stages=False`` to omit the per-stage pages and keep the report
    compact (the grid + Stage 8/9/10 + composite + outputs + filter_lab +
    color_atlas + hough are still there).

    Set ``filter_lab=False`` or ``color_atlas=False`` to drop the corresponding
    global page (e.g. for compact reports where only the lesion-focused outputs
    are wanted).

    Pass ``filename`` (e.g. ``"derm_report_2026-09-29_abc_a1b3c3d4e5f6.pdf"``)
    to override the on-disk name. Defaults to ``"report.pdf"`` to preserve
    the CLI's documented output layout.

    Missing files are skipped silently — the report adapts to whatever
    artifacts the run actually produced.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / (filename or "report.pdf")
    pdf_bytes = build_report_bytes(
        out,
        summary,
        hough=hough,
        all_outputs=all_outputs,
        expand_stages=expand_stages,
        filter_lab=filter_lab,
        color_atlas=color_atlas,
    )
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def build_report_bytes(
    output_dir: str | Path,
    summary: dict,
    *,
    hough: bool = False,
    all_outputs: bool = False,
    expand_stages: bool = True,
    filter_lab: bool = True,
    color_atlas: bool = True,
) -> bytes:
    """Same content as :func:`build_report_pdf`, but return raw PDF bytes.

    Useful for HTTP endpoints that want to stream the PDF directly without ever
    touching disk — the caller passes in a directory of PNG artifacts that
    already exist (e.g. a ``tempfile.TemporaryDirectory``) and gets back the
    encoded PDF.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _cover_page(pdf, summary, hough=hough)

        # The 2x4 grid is ALWAYS included — it preserves the original Colab
        # cell-11 style ("1. Original", "2. Grayscale", ...) that the user
        # explicitly wants to keep.
        grid_path = out / GRID_PAGE[0]
        if grid_path.is_file():
            _embed_image_page(pdf, grid_path, title=GRID_PAGE[1], cmap=GRID_PAGE[2])

        if expand_stages:
            for name, title, cmap in STAGE_PAGES:
                p = out / name
                if p.is_file():
                    _embed_image_page(pdf, p, title=title, cmap=cmap)

        # Stage 8/9/10 analysis pages — inserted between the preprocessing
        # stages and the post-detection outputs so the report flows from
        # "image cleanup" → "measurement" → "use of measurements".
        for name, title, cmap in ANALYSIS_PAGES:
            p = out / name
            if p.is_file():
                _embed_image_page(pdf, p, title=title, cmap=cmap)

        # Stage 11–15 ABCDE pages — the new clinical-measurement block.
        # Inserted after the legacy ANALYSIS_PAGES (8/9/10) and before the
        # post-detection outputs so the report reads "preprocess →
        # measure → clinical scoring → display".
        for name, title, cmap in ABCDE_PAGES:
            p = out / name
            if p.is_file():
                _embed_image_page(pdf, p, title=title, cmap=cmap)

        if all_outputs:
            p = out / ALL_PAGE[0]
            if p.is_file():
                _embed_image_page(pdf, p, title=ALL_PAGE[1], cmap=ALL_PAGE[2])

        for name, title, cmap in OUTPUT_PAGES:
            p = out / name
            if p.is_file():
                _embed_image_page(pdf, p, title=title, cmap=cmap)

        # Global visual-reference pages (always-on by default; not gated on
        # lesion detection because they use the full image).
        if filter_lab:
            p = out / FILTER_LAB_PAGE[0]
            if p.is_file():
                _embed_image_page(pdf, p, title=FILTER_LAB_PAGE[1], cmap=FILTER_LAB_PAGE[2])
        if color_atlas:
            p = out / COLOR_ATLAS_PAGE[0]
            if p.is_file():
                _embed_image_page(pdf, p, title=COLOR_ATLAS_PAGE[1], cmap=COLOR_ATLAS_PAGE[2])

        if hough:
            p = out / HOUGH_PAGE[0]
            if p.is_file():
                _embed_image_page(pdf, p, title=HOUGH_PAGE[1], cmap=HOUGH_PAGE[2])

    return buf.getvalue()
