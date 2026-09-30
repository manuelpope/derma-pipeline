"""FastAPI server exposing the dermatoscopio pipeline over HTTP.

Endpoints:

* ``GET  /health``            — liveness probe.
* ``POST /analyze``           — upload an image + tune knobs, get back the
                                summary JSON and URLs to every PNG/CSV.
* ``POST /report``            — same as ``/analyze`` but returns a multi-page
                                PDF. **Default mode is ephemeral**: the run
                                artifacts and the PDF live in a
                                ``tempfile.TemporaryDirectory()`` and are
                                erased after the response is sent. Pass
                                ``keep_artifacts=true`` to fall back to the
                                permanent ``output_root/<run_id>/`` directory.
* ``GET  /outputs/{run_id}/...`` — serve the artifacts of a previous run
                                  (only when ``keep_artifacts=true``).
* ``GET  /``                  — tiny index pointing at ``/docs`` (Swagger UI).
* ``GET  /docs``              — Swagger UI (FastAPI built-in).

The FastAPI app deliberately reuses :func:`derm_pipeline.cli.run` so server
behavior matches the CLI 1:1. Run IDs are random hex strings; each call gets
its own directory under ``output_root/<run_id>/`` (persistent mode) or in a
private tempdir (ephemeral mode).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import matplotlib
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

# Force a headless backend before any pyplot import happens elsewhere.
matplotlib.use("Agg")

from derm_pipeline.cli import PipelineConfig, run  # noqa: E402
from derm_pipeline.io import REPORT_ID_PATTERN, build_report_filename  # noqa: E402
from derm_pipeline.report import build_report_bytes, build_report_pdf  # noqa: E402

log = logging.getLogger("derm_pipeline.api")


# ---------------------------------------------------------------------------
# Internal helpers (shared by /analyze and /report)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Params:
    """Mirrors the FastAPI form fields exposed by both endpoints.

    Building the dataclass once and threading it into the pipeline avoids the
    alternative of having two near-identical endpoint bodies with their own
    copy-pasted ``PipelineConfig`` construction.
    """

    scale_factor: float
    num_zooms: int
    min_area: int
    min_dist: int
    zoom_base_radius: int
    zoom_relative: bool
    hough: bool
    save_stages: bool
    dpi: int
    all_outputs: bool
    filter_lab: bool
    color_atlas: bool

    def to_config(self, input_path: Path, output_dir: Path) -> PipelineConfig:
        return PipelineConfig(
            input_path=input_path,
            output_dir=output_dir,
            scale_factor=float(self.scale_factor),
            num_zooms=int(self.num_zooms),
            min_area=int(self.min_area),
            min_dist=int(self.min_dist),
            zoom_base_radius=int(self.zoom_base_radius),
            zoom_relative=bool(self.zoom_relative),
            hough=bool(self.hough),
            save_stages=bool(self.save_stages),
            display=False,  # server is always headless
            dpi=int(self.dpi),
            quiet=True,
            all_outputs=bool(self.all_outputs),
            filter_lab=bool(self.filter_lab),
            color_atlas=bool(self.color_atlas),
        )


def _safe_rmtree(path: str | Path) -> None:
    """Best-effort rmtree used as a BackgroundTask after ephemeral runs.

    Errors are swallowed — by the time this runs the response has already been
    delivered to the client, so we never want a cleanup failure to surface as
    an exception in the log.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _execute_run(
    image: UploadFile,
    params: _Params,
    root: Path,
    *,
    run_id: str,
    include_report: bool,
    expand_stages: bool = True,
    report_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Persist the upload, run the pipeline, optionally build the PDF.

    Returns ``(run_dir, summary_dict)`` where ``summary_dict`` already has
    ``run_id`` and ``artifacts`` populated. When ``include_report`` is true,
    ``report_pdf.filename`` is the dated + id-stamped name written to disk
    (and exposed via ``/outputs/<run_id>/<filename>``).
    """
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(image.filename or "upload").suffix or ".png"
    tmp_path = run_dir / f"_input{suffix}"
    try:
        with tmp_path.open("wb") as fh:
            shutil.copyfileobj(image.file, fh)
    finally:
        try:
            image.file.close()
        except Exception:
            pass

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    final_input = run_dir / f"input{suffix}"
    cfg = params.to_config(tmp_path, run_dir)

    try:
        summary = run(cfg)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IOError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("pipeline failed for run_id=%s", run_id)
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
    finally:
        try:
            tmp_path.rename(final_input)
        except FileNotFoundError:
            pass

    if include_report:
        report_filename = build_report_filename(run_id, report_id)
        try:
            report_path = build_report_pdf(
                run_dir,
                summary,
                hough=params.hough,
                all_outputs=params.all_outputs,
                expand_stages=expand_stages,
                filename=report_filename,
            )
            summary["report_pdf"] = {
                "filename": report_filename,
                "path": str(report_path),
                "url": f"/outputs/{run_id}/{report_filename}",
            }
        except Exception as exc:
            log.exception("report build failed for run_id=%s", run_id)
            raise HTTPException(status_code=500, detail=f"report error: {exc}") from exc

    artifacts = sorted(
        p.name for p in run_dir.iterdir() if p.is_file() and not p.name.startswith("_")
    )
    summary = _jsonable(summary)
    summary["run_id"] = run_id
    summary["artifacts"] = [
        {"name": name, "url": f"/outputs/{run_id}/{name}"} for name in artifacts
    ]
    return run_dir, summary



# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(output_root: str | Path = "./api_outputs") -> FastAPI:
    """Construct a FastAPI app rooted at ``output_root``.

    Each /analyze call writes into ``output_root/<run_id>/`` and exposes
    files under ``/outputs/<run_id>/<filename>``.
    """
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="Dermatoscopio Pipeline API",
        version="0.1.0",
        description=(
            "Upload a dermatoscope image, tune the segmentation and zoom "
            "knobs, and get back annotated PNGs + per-lesion metrics."
        ),
    )
    app.state.output_root = root

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "output_root": str(root)}

    @app.post("/analyze")
    def analyze(
        image: Annotated[UploadFile, File(description="Image file (jpg/png/etc.)")],
        scale_factor: Annotated[float, Form(ge=0.1, le=20.0)] = 2.0,
        num_zooms: Annotated[int, Form(ge=1, le=10)] = 3,
        min_area: Annotated[int, Form(ge=0, le=1_000_000)] = 5000,
        min_dist: Annotated[int, Form(ge=10, le=2000)] = 300,
        zoom_base_radius: Annotated[int, Form(ge=20, le=2000)] = 240,
        zoom_relative: Annotated[bool, Form()] = False,
        hough: Annotated[bool, Form()] = False,
        save_stages: Annotated[bool, Form()] = True,
        dpi: Annotated[int, Form(ge=50, le=600)] = 150,
        all_outputs: Annotated[
            bool,
            Form(
                description=(
                    "Save a single 3x4 composite '99_all_outputs.png' with every "
                    "visualization, instead of one PNG per stage."
                )
            ),
        ] = False,
        filter_lab: Annotated[
            bool,
            Form(
                description=(
                    "If true (default), generate '15_filter_lab.png' with 16 filter "
                    "variants (bilateral / CLAHE / blur / edge) of the full image."
                )
            ),
        ] = True,
        color_atlas: Annotated[
            bool,
            Form(
                description=(
                    "If true (default), generate '16_color_atlas.png' with 16 views "
                    "in LAB / HSV / YCrCb / channel-difference spaces."
                )
            ),
        ] = True,
    ) -> JSONResponse:
        run_id = uuid.uuid4().hex[:12]
        params = _Params(
            scale_factor=scale_factor,
            num_zooms=num_zooms,
            min_area=min_area,
            min_dist=min_dist,
            zoom_base_radius=zoom_base_radius,
            zoom_relative=zoom_relative,
            hough=hough,
            save_stages=save_stages,
            dpi=dpi,
            all_outputs=all_outputs,
            filter_lab=filter_lab,
            color_atlas=color_atlas,
        )
        _, summary = _execute_run(image, params, root, run_id=run_id, include_report=False)
        return JSONResponse(summary)

    @app.post("/report")
    def report(
        image: Annotated[UploadFile, File(description="Image file (jpg/png/etc.)")],
        scale_factor: Annotated[float, Form(ge=0.1, le=20.0)] = 2.0,
        num_zooms: Annotated[int, Form(ge=1, le=10)] = 3,
        min_area: Annotated[int, Form(ge=0, le=1_000_000)] = 5000,
        min_dist: Annotated[int, Form(ge=10, le=2000)] = 300,
        zoom_base_radius: Annotated[int, Form(ge=20, le=2000)] = 240,
        zoom_relative: Annotated[bool, Form()] = False,
        hough: Annotated[bool, Form()] = False,
        save_stages: Annotated[bool, Form()] = True,
        dpi: Annotated[int, Form(ge=50, le=600)] = 150,
        all_outputs: Annotated[
            bool,
            Form(
                description=(
                    "If true, the report PDF also embeds the 3x4 '99_all_outputs.png' "
                    "composite overview."
                )
            ),
        ] = False,
        filter_lab: Annotated[
            bool,
            Form(
                description=(
                    "If true (default), include the 'Laboratorio de Filtros' page "
                    "in the PDF with 16 filter variants of the full image."
                )
            ),
        ] = True,
        color_atlas: Annotated[
            bool,
            Form(
                description=(
                    "If true (default), include the 'Atlas de Espacios de Color' "
                    "page in the PDF with 16 color-space / channel-mix views."
                )
            ),
        ] = True,
        keep_artifacts: Annotated[
            bool,
            Form(
                description=(
                    "If true (default false), artifacts are persisted under "
                    "``output_root/<run_id>/`` and the PDF is also served by "
                    "``GET /outputs/<run_id>/report.pdf``. If false, the run uses a "
                    "private TemporaryDirectory and the bytes are streamed back as "
                    "``application/pdf``; both the PNGs and the PDF are deleted as "
                    "soon as the response has been delivered."
                )
            ),
        ] = False,
        expand_stages: Annotated[
            bool,
            Form(
                description=(
                    "If true (default), each preprocessing stage (00_original → "
                    "08_components) becomes its own PDF page with a descriptive "
                    "title. Set false to fall back to a single 2x4 grid page "
                    "('00_pipeline_stages.png')."
                )
            ),
        ] = True,
        report_id: Annotated[
            str | None,
            Form(
                max_length=12,
                pattern=REPORT_ID_PATTERN,
                description=(
                    "Optional user-supplied id ([A-Za-z0-9], ≤12 chars). "
                    "Appears in the generated PDF filename between the date "
                    "and the random run_id."
                ),
            ),
        ] = None,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> Response:
        """Run the pipeline and stream ``report.pdf`` back as the response body.

        Default mode is **ephemeral**:

        * All intermediate PNGs are written into a private
          :class:`tempfile.TemporaryDirectory` (typically under
          ``/var/folders/...`` on macOS).
        * The PDF is built directly into a ``BytesIO`` buffer via
          :func:`derm_pipeline.report.build_report_bytes`; no PDF is ever written
          to disk.
        * A :class:`fastapi.BackgroundTasks` hook removes the temp directory once
          the response has been sent.

        Set ``keep_artifacts=true`` to switch to the legacy persistent mode (PDF
        and PNGs written under ``output_root/<run_id>/``, served by the
        ``/outputs/...`` endpoint afterwards).
        """
        params = _Params(
            scale_factor=scale_factor,
            num_zooms=num_zooms,
            min_area=min_area,
            min_dist=min_dist,
            zoom_base_radius=zoom_base_radius,
            zoom_relative=zoom_relative,
            hough=hough,
            save_stages=save_stages,
            dpi=dpi,
            all_outputs=all_outputs,
            filter_lab=filter_lab,
            color_atlas=color_atlas,
        )
        run_id = uuid.uuid4().hex[:12]
        report_filename = build_report_filename(run_id, report_id)

        if keep_artifacts:
            # ---- persistent path (legacy behavior) ----
            run_dir, summary = _execute_run(
                image,
                params,
                root,
                run_id=run_id,
                include_report=True,
                expand_stages=expand_stages,
                report_id=report_id,
            )
            pdf_path = run_dir / report_filename
            if not pdf_path.is_file():
                raise HTTPException(
                    status_code=500, detail=f"{report_filename} was not produced"
                )
            body = pdf_path.read_bytes()
        else:
            # ---- ephemeral path: nothing survives this request ----
            with tempfile.TemporaryDirectory(prefix="derm_report_") as tmp_root_str:
                tmp_root = Path(tmp_root_str)
                run_dir, summary = _execute_run(
                    image,
                    params,
                    tmp_root,
                    run_id=run_id,
                    include_report=False,
                    expand_stages=expand_stages,
                    report_id=report_id,
                )
                body = build_report_bytes(
                    run_dir,
                    summary,
                    hough=params.hough,
                    all_outputs=params.all_outputs,
                    expand_stages=expand_stages,
                    filter_lab=params.filter_lab,
                    color_atlas=params.color_atlas,
                )
                # Belt-and-suspenders cleanup: `with` already deletes on exit,
                # but BackgroundTasks runs *after* the response is flushed, which
                # gives the client one extra millisecond of cleanup safety.
                background_tasks.add_task(_safe_rmtree, tmp_root_str)
                # strip the artifact URLs from the response header — they point
                # at files that will not exist in 1ms
                summary.pop("artifacts", None)

        if not body:
            raise HTTPException(status_code=500, detail=f"{report_filename} is empty")

        return Response(
            content=body,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{report_filename}"',
                "X-Run-Id": run_id,
                "X-Report-Id": report_id or "",
                "X-Storage": "persistent" if keep_artifacts else "ephemeral",
                "X-Pdf-Size": str(len(body)),
            },
        )

    @app.get("/outputs/{run_id}/{filename}")
    def serve_output(run_id: str, filename: str) -> FileResponse:
        # basic traversal guard
        if "/" in filename or ".." in filename or "/" in run_id or ".." in run_id:
            raise HTTPException(status_code=400, detail="invalid path")
        path = root / run_id / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"{filename} not found in {run_id}")
        return FileResponse(path)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(obj):
    """Recursively coerce numpy/path values to JSON-friendly python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# Module-level WSGI/ASGI handle for `uvicorn derm_pipeline.api:app`
app = build_app()