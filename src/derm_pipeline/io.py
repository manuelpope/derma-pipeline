"""I/O helpers: image loading and output persistence.

Replaces the Colab-specific `google.colab.files.upload()` / `files.download()`
flow from the original notebook. All paths are explicit and absolute so the
pipeline works locally without a runtime-managed working directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray


def load_image(path: str | Path) -> NDArray[np.uint8]:
    """Read an image from disk and normalize it to a 3-channel BGR ``uint8`` ndarray.

    Replaces the `files.upload()` + keyword-match + `cv2.imread(selected_file)`
    block from notebook cell 3. The "first file containing '5179'" name-picking
    heuristic is intentionally dropped — it was specific to the original
    screenshot and has no place in a reusable pipeline.

    Args:
        path: File path to read. Grayscale and RGBA inputs are accepted and
            normalized to BGR so downstream code never has to branch on channels.

    Returns:
        ``uint8`` ndarray of shape ``(H, W, 3)`` in OpenCV's BGR channel order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IOError: If OpenCV cannot decode the file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input image not found: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV failed to decode image: {p}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[-1] == 3:
        pass  # already BGR
    else:
        raise IOError(f"unsupported channel count {img.shape[-1]} in {p}")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def stage_filename(index: int, label: str, ext: str = "png") -> str:
    """Build the output filename for a numbered pipeline stage.

    Centralizing this keeps the output map a single source of truth, so
    renaming a stage doesn't require touching call sites.

    Args:
        index: Zero-padded sequence number (e.g. ``1`` -> ``"01"``).
        label: Short stage label, e.g. ``"gray"``.
        ext: File extension without the dot.

    Returns:
        Filename string like ``"01_gray.png"``.
    """
    return f"{index:02d}_{label}.{ext}"


# Pattern used by the FastAPI Form(...) field for the optional user-supplied
# report id. Kept here so the endpoint signature stays a one-liner.
REPORT_ID_PATTERN: str = r"^[A-Za-z0-9]{1,12}$"


def build_report_filename(
    run_id: str,
    report_id: str | None = None,
    *,
    when: datetime | None = None,
) -> str:
    """Return e.g. ``derm_report_2026-09-29_a1b3c3d4e5f6.pdf``.

    Format: ``derm_report_<YYYY-MM-DD>[_<report_id>]_<run_id>.pdf``.
    The ``run_id`` is always appended to guarantee uniqueness when two
    callers use the same ``report_id`` in the same UTC second. ``report_id``
    is assumed pre-validated against :data:`REPORT_ID_PATTERN` (alnum,
    ≤12 chars). ``when`` defaults to UTC ``now``; pass for tests.
    """
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    if report_id:
        return f"derm_report_{stamp}_{report_id}_{run_id}.pdf"
    return f"derm_report_{stamp}_{run_id}.pdf"


def save_outputs(
    stages: dict[str, NDArray[np.uint8] | None],
    output_dir: str | Path,
    *,
    save_rgb: Iterable[str] = (),
) -> list[Path]:
    """Persist a dict of named pipeline stages into ``output_dir``.

    Single-channel stages are written as-is. Stages listed in ``save_rgb``
    are converted RGB→BGR before writing so they look right in image viewers.
    Returns the list of files actually written, in insertion order.

    Args:
        stages: Mapping of filename -> ndarray. ``None`` values are skipped.
        output_dir: Directory to create (parents included) and write into.
        save_rgb: Filenames (no extension) whose content is in RGB and must be
            converted to BGR before `cv2.imwrite`.

    Returns:
        List of paths written, one per non-None entry in ``stages``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb_set = set(save_rgb)
    written: list[Path] = []

    for name, arr in stages.items():
        if arr is None:
            continue
        path = out / name
        payload = arr
        if name.split(".")[0] in rgb_set:
            # arr is RGB; OpenCV wants BGR for PNG fidelity
            payload = cv2.cvtColor(payload, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), payload)
        if not ok:
            raise IOError(f"cv2.imwrite failed for {path}")
        written.append(path)

    return written


def write_text(path: str | Path, content: str) -> Path:
    """Write ``content`` to ``path`` as UTF-8 text. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p