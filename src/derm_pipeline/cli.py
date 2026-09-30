"""Command-line interface for the dermatoscopio pipeline.

Two subcommands:

* ``run``    — process a single image: load → preprocess → segment → zooms.
* ``serve`` — launch a FastAPI server exposing the same pipeline over HTTP.

The ``run`` orchestrator is what the FastAPI layer also calls, so behavior
matches between the two entry points.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from derm_pipeline.borders import (
    crop_radius_for,
    crop_zooms,
    detect_hough_circles,
    find_critical_points,
)
from derm_pipeline.enhancement import enhance_color_lab
from derm_pipeline.io import load_image, save_outputs, stage_filename, write_text
from derm_pipeline.metrics import compute_metrics, df_to_jsonable, lesions_to_df, save_csv
from derm_pipeline.preprocess import preprocess
from derm_pipeline.report import (
    COLOR_ATLAS_PAGE,
    FILTER_LAB_PAGE,
    build_report_pdf,
)
from derm_pipeline.segmentation import find_lesions, lesion_contours, lesion_mask
from derm_pipeline.viz import (
    plot_all_outputs,
    plot_asymmetry,
    plot_border_shape,
    plot_color_analysis,
    plot_color_space_atlas,
    plot_contour_overlay,
    plot_contour_topography,
    plot_critical_zooms,
    plot_filter_lab,
    plot_hough_circles,
    plot_lesion_segmentation,
    plot_lesions,
    plot_main_lesion_zoom,
    plot_pipeline_stages,
)

log = logging.getLogger("derm_pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved, immutable settings for a single pipeline run."""

    input_path: Path
    output_dir: Path
    scale_factor: float = 2.0
    num_zooms: int = 3
    min_area: int = 5000
    min_dist: int = 300
    zoom_base_radius: int = 240
    zoom_relative: bool = False
    hough: bool = False
    save_stages: bool = True
    display: bool = False
    dpi: int = 150
    quiet: bool = False
    all_outputs: bool = False
    report: bool = False
    compact_report: bool = False
    filter_lab: bool = True
    color_atlas: bool = True

    def summary(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", type=Path, required=True, help="path to the input image")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("./output"))
    p.add_argument(
        "--scale-factor",
        type=float,
        default=2.0,
        help="zoom multiplier; 2.0 = 2x more detail (tighter crop), 0.5 = more context",
    )
    p.add_argument("--num-zooms", type=int, default=3)
    p.add_argument("--min-area", type=int, default=5000)
    p.add_argument("--min-dist", type=int, default=300)
    p.add_argument("--zoom-base-radius", type=int, default=240)
    p.add_argument(
        "--zoom-relative",
        action="store_true",
        help="set base radius = 0.4 * min(lesion_w, lesion_h) instead of the fixed pixel value",
    )
    p.add_argument("--hough", action="store_true", help="enable Hough circle detection")
    p.add_argument("--no-save-stages", dest="save_stages", action="store_false")
    p.add_argument("--display", action="store_true", help="try to pop a window after saving")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--all",
        dest="all_outputs",
        action="store_true",
        help="also save a single 3x6 composite '99_all_outputs.png' with every visualization",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="after the run, build a multi-page report.pdf inside --output-dir",
    )
    p.add_argument(
        "--compact-report",
        action="store_true",
        help="compress each preprocessing stage into a single 2x4 grid page (default: every stage gets its own page)",
    )
    p.add_argument(
        "--filter-lab",
        dest="filter_lab",
        action="store_true",
        default=True,
        help="agrega '18_filter_lab.png' con 16 variantes de filtro (bilateral, CLAHE, blur, bordes) (default: true)",
    )
    p.add_argument("--no-filter-lab", dest="filter_lab", action="store_false")
    p.add_argument(
        "--color-atlas",
        dest="color_atlas",
        action="store_true",
        default=True,
        help="agrega '19_color_atlas.png' con vistas en LAB / HSV / YCrCb / diferencias de canal (default: true)",
    )
    p.add_argument("--no-color-atlas", dest="color_atlas", action="store_false")


def _add_serve_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("./api_outputs"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="derm-pipeline",
        description="Local dermatological image processing pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="process a single image")
    _add_run_args(p_run)

    p_serve = sub.add_parser("serve", help="start the FastAPI HTTP server")
    _add_serve_args(p_serve)

    return parser


# ---------------------------------------------------------------------------
# Backend selection — headless by default, opt-in window
# ---------------------------------------------------------------------------


def _select_matplotlib_backend(display: bool) -> None:
    """Pick a matplotlib backend. `Agg` is set unconditionally first to avoid
    "Agg already has a figure" warnings when the process is reused."""
    matplotlib.use("Agg")
    if display:
        # macOS first; TkAgg as a portable fallback. Linux/WSL generally
        # have TkAgg; if neither loads, we silently keep Agg.
        for candidate in ("MacOSX", "TkAgg", "QtAgg"):
            try:
                matplotlib.use(candidate, force=True)
                return
            except Exception:
                continue


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def run(config: PipelineConfig) -> dict:
    """Execute the full pipeline. Returns a JSON-serializable summary dict.

    Side effects:
        * writes PNGs and CSVs into ``config.output_dir``
        * writes ``pipeline_summary.json`` with run metadata
        * optionally pops a matplotlib window if ``config.display`` is True
    """
    warnings.simplefilter("default", UserWarning)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    bgr = load_image(config.input_path)

    # 2. Preprocess (notebook cell 9)
    stages = preprocess(bgr)

    # 3. Lesion segmentation (cell 13)
    lesions = find_lesions(
        stages["labels"],
        stages["num_labels"],
        bgr.shape,
        stats=stages["cc_stats"],
        centroids=stages["cc_centroids"],
        min_area=config.min_area,
    )
    summary = {
        "input": str(config.input_path),
        "image_shape": list(bgr.shape),
        "scale_factor": config.scale_factor,
        "num_zooms_requested": config.num_zooms,
        "min_area": config.min_area,
        "zoom_base_radius": config.zoom_base_radius,
        "num_lesions": len(lesions),
        "lesion_details": df_to_jsonable(lesions_to_df(lesions)),
        "warnings": [],
    }

    # 4. Persist the 6 stage images that are visible in the final report
    #    (Stages 6, 7 and 8 are still in `stages` for downstream lesion
    #    detection but are not written out as PNG artifacts.)
    if config.save_stages:
        stage_imgs = {
            stage_filename(0, "original"): stages["rgb"],  # 0_original.png (RGB)
            stage_filename(1, "gray"): stages["gray"],
            stage_filename(2, "bilateral"): stages["bilateral"],
            stage_filename(3, "blur"): stages["blur"],
            stage_filename(4, "enhanced"): stages["enhanced"],
            stage_filename(5, "normalized"): stages["normalized"],
        }
        save_outputs(
            stage_imgs,
            config.output_dir,
            save_rgb=("00_original",),
        )
        plot_pipeline_stages(
            stages,
            save_path=config.output_dir / "00_pipeline_stages.png",
            show=config.display,
            dpi=config.dpi,
        )

        # 4.5 Global visual-reference pages (filter lab + color atlas).
        # Independent of lesion detection — they consume only the RGB image,
        # so they're generated before find_lesions so they always appear.
        if config.filter_lab:
            plot_filter_lab(
                stages["rgb"],
                save_path=config.output_dir / "18_filter_lab.png",
                show=config.display,
                dpi=config.dpi,
            )
        if config.color_atlas:
            plot_color_space_atlas(
                stages["rgb"],
                save_path=config.output_dir / "19_color_atlas.png",
                show=config.display,
                dpi=config.dpi,
            )

    # 5. Metrics + per-lesion overlays
    metrics_df = compute_metrics(lesions, stages["labels"], rgb=stages["rgb"])
    if metrics_df.empty:
        msg = "no lesions detected above min_area; skipping lesion-dependent stages"
        warnings.warn(msg, UserWarning)
        summary["warnings"].append(msg)
        # still write empty CSVs and a JSON summary
        save_csv(metrics_df, config.output_dir / "lesions.csv")
        save_csv(metrics_df, config.output_dir / "metrics.csv")
        write_text(
            config.output_dir / "pipeline_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False),
        )
        return summary

    save_csv(lesions_to_df(lesions), config.output_dir / "lesions.csv")
    save_csv(metrics_df, config.output_dir / "metrics.csv")
    summary["metrics"] = df_to_jsonable(metrics_df)

    # --- Stage 8 — Lesion Segmentation ---
    plot_lesion_segmentation(
        stages["rgb"],
        lesions,
        stages["labels"],
        save_path=config.output_dir / "08_segmentation.png",
        show=config.display,
        dpi=config.dpi,
    )

    # --- Stage 9 — Color Analysis ---
    plot_color_analysis(
        stages["rgb"],
        lesions,
        stages["labels"],
        save_path=config.output_dir / "09_color_analysis.png",
        show=config.display,
        dpi=config.dpi,
    )

    # annotated detections (post-segmentation output)
    plot_lesions(
        stages["rgb"],
        lesions,
        save_path=config.output_dir / "12_detected.png",
        show=config.display,
        dpi=config.dpi,
    )

    # main lesion zoom + boundary (cell 15)
    main_lesion = max(lesions, key=lambda d: d["area"])
    mask = lesion_mask(stages["labels"], main_lesion["id"])
    plot_main_lesion_zoom(
        stages["rgb"],
        main_lesion,
        stages["labels"],
        save_path=config.output_dir / "13_main_lesion_zoom.png",
        show=config.display,
        dpi=config.dpi,
    )

    # contour topography (cell 17)
    x, y, w, h = main_lesion["x"], main_lesion["y"], main_lesion["width"], main_lesion["height"]
    rgb_roi = stages["rgb"][y : y + h, x : x + w]
    gray_roi = stages["gray"][y : y + h, x : x + w]
    plot_contour_topography(
        rgb_roi,
        gray_roi,
        save_path=config.output_dir / "14_topography.png",
        show=config.display,
        dpi=config.dpi,
    )

    # magenta silhouettes of every lesion (cell 19)
    all_contours = []
    for obj in lesions:
        comp_mask = lesion_mask(stages["labels"], obj["id"])
        _, cs = lesion_contours(comp_mask)
        if cs:
            all_contours.append(max(cs, key=cv2.contourArea))
    plot_contour_overlay(
        stages["rgb"],
        all_contours,
        save_path=config.output_dir / "15_contours_only.png",
        show=config.display,
        dpi=config.dpi,
    )

    # 6. Critical border zooms with corrected scale_factor semantics (cell 21/23)
    enhanced_rgb = enhance_color_lab(stages["rgb"])
    main_mask = lesion_mask(stages["labels"], main_lesion["id"])
    biggest_contour, _ = lesion_contours(main_mask)
    if biggest_contour is None:
        msg = "no contour on main lesion; skipping zoom patches"
        warnings.warn(msg, UserWarning)
        summary["warnings"].append(msg)
    else:
        # resolve base_radius
        base_radius = config.zoom_base_radius
        if config.zoom_relative:
            base_radius = max(40, int(0.4 * min(main_lesion["width"], main_lesion["height"])))
            summary["zoom_base_radius_effective"] = base_radius

        points = find_critical_points(
            enhanced_rgb,
            main_mask,
            n=config.num_zooms,
            min_dist=config.min_dist,
        )
        summary["zoom_points"] = [list(p) for p in points]
        summary["num_zooms_placed"] = len(points)

        if not points:
            msg = "no critical points found on lesion border"
            warnings.warn(msg, UserWarning)
            summary["warnings"].append(msg)
        else:
            patches = crop_zooms(
                enhanced_rgb,
                points,
                scale_factor=config.scale_factor,
                base_radius=base_radius,
            )
            crop_radius = crop_radius_for(base_radius, config.scale_factor)
            summary["crop_radius_pixels"] = crop_radius
            summary["patch_size"] = [2 * crop_radius, 2 * crop_radius]

            plot_critical_zooms(
                enhanced_rgb,
                points,
                patches,
                contour=biggest_contour,
                crop_radius=crop_radius,
                save_path=config.output_dir / "16_critical_zooms.png",
                show=config.display,
                dpi=config.dpi,
            )

    # --- Stage 10 — Border & Shape Analysis ---
    # Always rendered when there is at least one lesion; consumes the same
    # labels as the rest of the analysis so the metrics stay consistent.
    plot_border_shape(
        stages["rgb"],
        lesions,
        stages["labels"],
        save_path=config.output_dir / "10_border_shape.png",
        show=config.display,
        dpi=config.dpi,
    )

    # --- Stage 11 — 2-axis asymmetry (ABCD-A) ---
    # (Stage 10 — Diameter (ABCD-D) — was removed per user feedback;
    #  diameter values are still in metrics.csv.)
    plot_asymmetry(
        stages["rgb"],
        lesions,
        stages["labels"],
        save_path=config.output_dir / "11_asymmetry.png",
        show=config.display,
        dpi=config.dpi,
    )

    # 7. Hough circles — opt-in
    circles: np.ndarray | None = None
    if config.hough:
        circles = detect_hough_circles(stages["enhanced"])
        summary["hough_circles"] = (
            [[int(x), int(y), int(r)] for x, y, r in circles] if circles is not None else []
        )
        plot_hough_circles(
            stages["rgb"],
            circles,
            save_path=config.output_dir / "17_hough_circles.png",
            show=config.display,
            dpi=config.dpi,
        )

    # 8. Composite overview (opt-in via --all / all=True).
    # Falls back gracefully when earlier stages had no lesions / no critical
    # points / hough not enabled.
    if config.all_outputs:
        # pull values computed inside the contour block; may be unset when no
        # critical points were found
        try:
            final_points = points  # noqa: F821
            final_patches = patches  # noqa: F821
            final_crop_radius = crop_radius  # noqa: F821
        except NameError:
            final_points, final_patches, final_crop_radius = [], [], 0

        plot_all_outputs(
            stages["rgb"],
            stages,
            lesions,
            main_lesion,
            main_mask,
            rgb_roi,
            gray_roi,
            all_contours,
            enhanced_rgb,
            biggest_contour,
            final_points,
            final_patches,
            final_crop_radius,
            circles,
            labels=stages["labels"],
            save_path=config.output_dir / "99_all_outputs.png",
            show=config.display,
            dpi=config.dpi,
        )

    # 8. JSON summary last so any warnings collected above are included
    write_text(
        config.output_dir / "pipeline_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )

    # 9. Optional PDF report (--report). Always persisted on disk; the
    # FastAPI /report endpoint reads the bytes back to return them binary.
    if config.report:
        report_path = build_report_pdf(
            config.output_dir,
            summary,
            hough=config.hough,
            all_outputs=config.all_outputs,
            expand_stages=not config.compact_report,
            filter_lab=config.filter_lab,
            color_atlas=config.color_atlas,
        )
        if not config.quiet:
            log.info("report written to %s", report_path)
        summary["report_pdf"] = str(report_path)

    if not config.quiet:
        log.info(
            "done: %d lesions, %d zoom points (scale_factor=%.2f, crop_radius=%dpx)",
            len(lesions),
            summary.get("num_zooms_placed", 0),
            config.scale_factor,
            summary.get("crop_radius_pixels", 0),
        )
        log.info("outputs written to %s", config.output_dir)

    return summary


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        scale_factor=float(args.scale_factor),
        num_zooms=int(args.num_zooms),
        min_area=int(args.min_area),
        min_dist=int(args.min_dist),
        zoom_base_radius=int(args.zoom_base_radius),
        zoom_relative=bool(args.zoom_relative),
        hough=bool(args.hough),
        save_stages=bool(args.save_stages),
        display=bool(args.display),
        dpi=int(args.dpi),
        quiet=bool(args.quiet),
        all_outputs=bool(args.all_outputs),
        report=bool(args.report),
        compact_report=bool(args.compact_report),
        filter_lab=bool(args.filter_lab),
        color_atlas=bool(args.color_atlas),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    _select_matplotlib_backend(config.display)
    run(config)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    _select_matplotlib_backend(False)  # server is always headless
    # local import to keep CLI snappy when only `run` is used
    import uvicorn

    from derm_pipeline.api import build_app

    app = build_app(output_root=args.output_root)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())