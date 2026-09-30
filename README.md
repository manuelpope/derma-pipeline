# derm-pipeline

Local replacement for the original `img_filtros_dermatoscopio.ipynb` Colab
notebook. Same algorithms, no Colab runtime needed, no `/content/` paths.

## Install

```bash
cd ~/Downloads/derm_pipeline
uv sync
```

(uv creates `.venv/` and pins Python 3.11 in `.python-version`.)

## CLI usage

```bash
uv run derm-pipeline run \
    --input /path/to/skin_image.jpg \
    --output-dir ./out \
    --scale-factor 2.0 \
    --num-zooms 3
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `-i` | required | path to image file |
| `--output-dir`, `-o` | `./output` | where to write PNGs / CSVs |
| `--scale-factor` | `2.0` | **zoom multiplier**: `2.0` = 2× more detail (tighter physical crop); `1.0` = baseline; `0.5` = more context |
| `--num-zooms` | `3` | how many critical-border zooms to extract |
| `--min-area` | `5000` | lesion area threshold in pixels |
| `--min-dist` | `300` | NMS distance between zoom centers |
| `--zoom-base-radius` | `240` | physical crop radius when `scale-factor=1.0` |
| `--zoom-relative` | off | switch to `0.4 * min(lesion_w,h)` for the base |
| `--hough` | off | also run Hough circle detection |
| `--no-save-stages` | stages on | skip the per-stage PNGs |
| `--display` | off | pop a window after saving (macOS / Tk) |
| `--dpi` | `150` | matplotlib output DPI |
| `--quiet` | off | suppress INFO logs |

### Outputs

```text
output_dir/
├── 00_original.png              ← input as RGB
├── 00_pipeline_stages.png       ← 2×3 grid of the 6 preprocessing stages
├── 01_gray.png  …  05_normalized.png
├── 08_segmentation.png          ← Stage 8 — original / binary mask / colored overlay
├── 09_color_analysis.png        ← Stage 9 — per-lesion card · LARGE thumbnail · LAB mean · k-means dominant colors with frequency %
├── 10_border_shape.png          ← Stage 10 — compact metrics table · circularity · symmetry · irregularity per lesion (with green/yellow/red flag cell)
├── 11_asymmetry.png             ← Stage 11 — ABCD-A: original + fitted ellipse + red major axis + cyan minor axis + A₂ = h+v score badge
├── 12_detected.png              ← green bboxes + IDs
├── 13_main_lesion_zoom.png      ← ROI before/after boundary
├── 14_topography.png            ← spring contourf
├── 15_contours_only.png         ← magenta silhouettes
├── 16_critical_zooms.png        ← the cell 21/23 figure
├── 17_hough_circles.png         ← only if --hough
├── 18_filter_lab.png            ← 16-variant filter lab (always on)
├── 19_color_atlas.png           ← LAB / HSV / YCrCb atlas (always on)
├── 99_all_outputs.png           ← 3×5 composite (only if --all) — includes Stage 9 / 10 / 11 thumbnails
├── lesions.csv                  ← basic table (cell 25)
├── metrics.csv                  ← area · perim · circ · symmetry · irregularity ·
│                                  radial_std · mean_L/a/b/H/S/V ·
│                                  asymmetry_h/v/2axis · diameter_eq/major/large_flag ·
│                                  pct_<6 named colors> · n_colors_present ·
│                                  border_octant_score · tds_score · tds_class
└── pipeline_summary.json        ← run metadata + warnings
```

> Note: `pct_<color>`, `n_colors_present`, `tds_score`, and `tds_class` are still computed and persisted in `metrics.csv` even though their standalone visualization pages (formerly Stages 11, 14, 15) were removed — the data is derivable from the LAB pixels + per-axis scores and is useful for downstream analysis (custom alerts, research notebooks) without needing to re-run the pipeline.

> Note: numbering after Stage 5 was renumbered so that Stages 8/9/10 line up
> with the user's mental model: stages 6/7/8 are still computed internally by
> `preprocess.py` for downstream segmentation but are not surfaced as
> artifacts (the user removed them previously).
>
> **Tier 1 — ABCDE rule of dermoscopy (Stage 11)**: closes **A** (2-axis
> asymmetry) with a deterministic flag + 0–10 score, plus the fitted
> ellipse + major/minor axes for visual confirmation. **B** (border
> irregularity) is handled by Stage 10's compact table. **C**
> (named-color palette), **D** (diameter), and the **Total Dermoscopy
> Score** (TDS = A·1.3 + B·0.1 + C·0.5 + D·0.5) were trialled and removed:
> each was a visualization without a diagnostic determination, and the
> score badge + flag cell already convey the diagnostic tier without the
> extra pages. Their underlying numeric columns are still computed and
> persisted in `metrics.csv`. **E** (evolución) is explicitly out of scope
> — it requires temporal comparison. The 7-point checklist / pattern
> analysis structures (pigment network, blue-white veil, vascular pattern,
> …) are reserved for Tier 2.

## HTTP API

```bash
uv run derm-pipeline serve --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000/docs** for the Swagger UI.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness probe |
| `POST` | `/analyze` | multipart upload; returns JSON summary + artifact URLs |
| `POST` | `/report` | multipart upload; streams back a multi-page PDF |
| `GET` | `/outputs/{run_id}/{filename}` | serve a PNG / CSV from a previous run |

`POST /analyze` accepts the same knobs as the CLI as `Form` fields
(`scale_factor`, `num_zooms`, `min_area`, `min_dist`, `zoom_base_radius`,
`zoom_relative`, `hough`, `save_stages`, `dpi`) plus an `image` file part.

Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
    -F "image=@/path/to/skin.jpg" \
    -F "scale_factor=3.0" \
    -F "num_zooms=4" \
    -F "hough=true"
```

The response includes `run_id` and `artifacts[]` with `url`s like
`/outputs/<run_id>/20_critical_zooms.png`.

`POST /report` runs the same pipeline and returns a PDF. The filename sent in
the `Content-Disposition` header is always dated:

```text
derm_report_<YYYY-MM-DD>[_<report_id>]_<run_id>.pdf
```

`report_id` is an **optional** form field (alphanumeric, ≤12 chars) — when
omitted the random `run_id` (12-hex) is used in its place. The `run_id` is
always appended so two concurrent requests with the same `report_id` never
collide. The same name is used on disk in `keep_artifacts=true` mode, so the
`/outputs/<run_id>/<…>` URL keeps in sync with the downloaded filename.

Response headers worth knowing:

| Header | Meaning |
|---|---|
| `Content-Disposition` | the dated filename above |
| `X-Run-Id` | the random hex `run_id` (used as directory name in persistent mode) |
| `X-Report-Id` | echo of the user-supplied id, or empty |
| `X-Storage` | `persistent` or `ephemeral` |
| `X-Pdf-Size` | byte length of the response body |

Example with `curl`:

```bash
# default → dated + run_id in the filename
curl -X POST http://127.0.0.1:8000/report \
    -F "image=@/path/to/skin.jpg" \
    -F "scale_factor=3.0" \
    -o report.pdf
# → report.pdf will be served as e.g. derm_report_2026-09-29_a1b3c3d4e5f6.pdf

# with a custom id → dated + id + run_id
curl -X POST http://127.0.0.1:8000/report \
    -F "image=@/path/to/skin.jpg" \
    -F "report_id=patient42" \
    -o report.pdf
# → report.pdf will be served as e.g. derm_report_2026-09-29_patient42_a1b3c3d4e5f6.pdf
```

> Note: persistent-mode URLs from before this change ended in
> `/outputs/<run_id>/report.pdf`. After upgrading, the on-disk filename is the
> dated one above, so any pre-existing bookmarks or scripts must be updated.

## Use as a Python module

```python
from derm_pipeline import (
    load_image, preprocess, find_lesions,
    enhance_color_lab, find_critical_points, crop_zooms,
)

bgr = load_image("skin.jpg")
stages = preprocess(bgr)
lesions = find_lesions(stages["labels"], stages["num_labels"], bgr.shape)
enhanced = enhance_color_lab(stages["rgb"])
mask = (stages["labels"] == lesions[0]["id"]).astype("uint8") * 255
points = find_critical_points(enhanced, mask, n=3, min_dist=300)
patches = crop_zooms(enhanced, points, scale_factor=2.0, base_radius=240)
```

## Docker / docker compose

A reproducible container image is provided — no local Python install required.

### Build + run with docker compose

```bash
docker compose up --build
# → service on http://localhost:8000  (Swagger UI at /docs)
```

Persistent artifacts from `keep_artifacts=true` runs are stored in the
`derm-pipeline-outputs` named volume and survive container restarts. Override
the host port with `DERM_PORT=9000 docker compose up`.

### Plain docker

```bash
docker build -t derm-pipeline:local .
docker run --rm -p 8000:8000 derm-pipeline:local
```

### Inspect what's inside

```bash
docker compose exec derm-api python -c "from derm_pipeline import api; print(api.app)"
docker compose logs -f
```

The image is ~700 MB (Debian slim + Python 3.11 + opencv-python + matplotlib).
The `Dockerfile` installs `uv` from `ghcr.io/astral-sh/uv` and resolves the
exact dep set pinned in `uv.lock`.

## `scale_factor` semantics (the bit that changed)

The original notebook used a `zoom_radius` parameter measured in pixels, and
its semantics were inverted — increasing `zoom_radius` showed *more* context,
which is the opposite of what "zoom" usually means.

Here, the formula is

```python
crop_radius = max(5, round(base_radius / scale_factor))
```

| scale_factor | base_radius=240 | physical patch | meaning |
|---|---|---|---|
| 0.5 | 480 | 960×960 | 2× more context |
| 1.0 | 240 | 480×480 | baseline |
| 2.0 | 120 | 240×240 | 2× more detail |
| 4.0 | 60  | 120×120 | 4× more detail |

Patches are *not* upsampled — the matplotlib axis renders them at the same
display size regardless, so detail per pixel scales with `scale_factor`.

## Notes

* The original `.ipynb` is preserved at `~/Downloads/img_filtros_dermatoscopio.ipynb`
  as historical reference; this package replaces it for ongoing work.
* The "first file containing '5179'" name-match heuristic from the notebook
  was a one-off for the original screenshot and is intentionally dropped.