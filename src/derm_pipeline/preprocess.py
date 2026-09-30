"""Image preprocessing pipeline.

Mirrors notebook cell 9 verbatim in algorithm, but returns a single ``dict``
instead of leaking ~10 globals. Every downstream module consumes this dict, so
its key set is the public contract of the package.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def preprocess(
    bgr: NDArray[np.uint8],
    *,
    bilateral_d: int = 15,
    bilateral_sigma_color: int = 75,
    bilateral_sigma_space: int = 75,
    blur_kernel: int = 15,
    clahe_clip_limit: float = 1.5,
    clahe_grid: tuple[int, int] = (8, 8),
    close_kernel: int = 15,
    open_kernel: int = 21,
) -> dict[str, NDArray]:
    """Run the full 8-stage preprocessing pipeline.

    Stages (matching the original notebook cell 9):

        BGR -> grayscale -> bilateral filter -> Gaussian blur
            -> CLAHE -> normalize -> Otsu (inverted) -> morphology (close+open)
            -> connected components

    Args:
        bgr: Input image as ``uint8`` BGR ndarray of shape ``(H, W, 3)``.
        bilateral_d: Diameter of the bilateral filter pixel neighborhood.
        bilateral_sigma_color: Filter sigma in the color space.
        bilateral_sigma_space: Filter sigma in the coordinate space.
        blur_kernel: Gaussian blur kernel size (must be odd).
        clahe_clip_limit: Contrast limit for CLAHE.
        clahe_grid: CLAHE tile grid size.
        close_kernel: Closing kernel (morphology) to fill holes.
        open_kernel: Opening kernel (morphology) to clear noise.

    Returns:
        Dict with keys:
            ``bgr``         original BGR input (echoed for downstream)
            ``rgb``         same image in RGB
            ``gray``        grayscale
            ``bilateral``   bilateral-filtered grayscale
            ``blur``        Gaussian-blurred bilateral
            ``enhanced``    CLAHE-enhanced grayscale
            ``normalized``  min/max normalized grayscale
            ``binary``      Otsu binary (inverted)
            ``closed``      morphologically closed binary
            ``clean``       morphologically opened after closing
            ``labels``      int32 connected-component label map
            ``num_labels``  int — number of labels (including background 0)
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(
        gray, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
    )

    blur = cv2.GaussianBlur(bilateral, (blur_kernel, blur_kernel), 0)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid)
    enhanced = clahe.apply(blur)

    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)

    return {
        "bgr": bgr,
        "rgb": rgb,
        "gray": gray,
        "bilateral": bilateral,
        "blur": blur,
        "enhanced": enhanced,
        "normalized": normalized,
        "binary": binary,
        "closed": closed,
        "clean": clean,
        "labels": labels,
        "num_labels": int(num_labels),
        "cc_stats": stats,
        "cc_centroids": centroids,
    }