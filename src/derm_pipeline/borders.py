"""Critical-border point detection and zoom cropping.

This module owns the ``scale_factor`` semantics that the user explicitly
asked for. The notebook's original `zoom_radius` parameter was a *physical*
crop radius whose semantics were inverted (increasing it added context, not
zoom). Here we fix that:

    crop_radius = max(MIN_RADIUS, round(base_radius / scale_factor))

So ``scale_factor=2.0`` means **2× more zoom** (tighter physical crop), which
matches clinical intuition. Patches are returned at their natural resolution
— the matplotlib axis renders them at the same display size regardless of
scale_factor, so detail per pixel goes up when scale_factor goes up.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


MIN_RADIUS: int = 5
"""Lower clamp on the physical crop radius — anything smaller would be a single
handful of pixels and uninformative."""


def find_critical_points(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    n: int = 3,
    *,
    min_dist: int = 300,
    candidate_step_divisor: int = 100,
    neighborhood_half: int = 10,
    grad_weight: float = 0.5,
    color_weight: float = 0.5,
    margin: int = 40,
) -> list[tuple[int, int]]:
    """Pick the ``n`` most clinically interesting points on a lesion's border.

    Mirrors notebook cell 21 steps 2–5:

        1. Find biggest external contour of ``mask``.
        2. Sobel gradient magnitude on the grayscale of ``rgb``.
        3. Sample up to ~100 candidates along the contour, score each as
           ``grad_weight * grad_mag + color_weight * std(neighborhood)``.
        4. Greedy NMS with ``min_dist`` separation in pixels.
        5. Margin filter: drop candidates within ``margin`` of the image edge.

    Args:
        rgb: Source image (RGB ``uint8``) — used for color contrast scoring.
        mask: Binary lesion mask (``uint8`` 0/255).
        n: Maximum number of points to return.
        min_dist: Minimum pixel distance between two selected points.
        candidate_step_divisor: Number of candidates sampled per contour
            length = ``len(contour) // candidate_step_divisor``.
        neighborhood_half: Half-size of the window used to compute color
            contrast (std) around each candidate.
        grad_weight: Weight of the gradient-magnitude term in the score.
        color_weight: Weight of the color-contrast term in the score.
        margin: Edge margin; candidates within this many pixels of any border
            are dropped.

    Returns:
        Ordered list of ``(x, y)`` tuples, length ``<= n``. Empty if the mask
        has no contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    h, w = rgb.shape[:2]
    if h < 2 * margin + 1 or w < 2 * margin + 1:
        # image too small for the margin — silently expand the safety zone
        margin = max(0, min(h, w) // 4)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # edge clamp on min_dist for very small images
    min_dist_eff = int(max(20, min(min_dist, min(h, w) // 3)))

    step = max(1, len(contour) // max(1, candidate_step_divisor))
    nh = int(neighborhood_half)
    candidates: list[tuple[int, int, float]] = []

    for i_pt in range(0, len(contour), step):
        pt = contour[i_pt][0]
        px, py = int(pt[0]), int(pt[1])

        if px < margin or px > w - margin or py < margin or py > h - margin:
            continue

        grad_val = float(grad_magnitude[py, px])
        y0, y1 = max(0, py - nh), min(h, py + nh)
        x0, x1 = max(0, px - nh), min(w, px + nh)
        color_contrast = float(np.std(rgb[y0:y1, x0:x1]))

        score = grad_weight * grad_val + color_weight * color_contrast
        candidates.append((px, py, score))

    candidates.sort(key=lambda c: c[2], reverse=True)

    selected: list[tuple[int, int]] = []
    for px, py, _ in candidates:
        if any((px - sx) ** 2 + (py - sy) ** 2 < min_dist_eff ** 2 for sx, sy in selected):
            continue
        selected.append((px, py))
        if len(selected) >= n:
            break

    return selected


def crop_radius_for(
    base_radius: int,
    scale_factor: float,
    *,
    min_radius: int = MIN_RADIUS,
) -> int:
    """Resolve the physical crop radius from ``scale_factor``.

    ``crop_radius = max(min_radius, round(base_radius / scale_factor))``.

    Emits a ``UserWarning`` once when the floor kicks in.
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}")

    raw = int(round(base_radius / scale_factor))
    if raw < min_radius:
        warnings.warn(
            f"requested scale_factor={scale_factor} yields crop_radius={raw} "
            f"(< MIN_RADIUS={min_radius}); clamping to {min_radius}",
            UserWarning,
            stacklevel=2,
        )
        return min_radius
    return raw


def crop_zooms(
    rgb: NDArray[np.uint8],
    points: Sequence[tuple[int, int]],
    *,
    scale_factor: float = 2.0,
    base_radius: int = 240,
) -> list[NDArray[np.uint8]]:
    """Crop square patches around each ``(x, y)`` point using ``scale_factor``.

    Patch size is ``2*crop_radius × 2*crop_radius`` pixels with the actual
    radius coming from :func:`crop_radius_for`. Coordinates are clamped to the
    image bounds — patches near an edge will simply be smaller than nominal.

    Args:
        rgb: Source image (RGB ``uint8``).
        points: Centers for each patch.
        scale_factor: Zoom multiplier. ``1.0`` = baseline (= ``base_radius``
            physical pixels). ``2.0`` = 2× more detail (half the patch).
            ``0.5`` = half the zoom (twice the context).
        base_radius: Physical crop radius in pixels when ``scale_factor=1.0``.

    Returns:
        List of RGB patches, in the same order as ``points``. May be empty.
    """
    if not points:
        return []

    crop_radius = crop_radius_for(base_radius, scale_factor)
    h, w = rgb.shape[:2]

    patches: list[NDArray[np.uint8]] = []
    for cx, cy in points:
        y1, y2 = max(0, cy - crop_radius), min(h, cy + crop_radius)
        x1, x2 = max(0, cx - crop_radius), min(w, cx + crop_radius)
        patches.append(rgb[y1:y2, x1:x2].copy())
    return patches


def detect_hough_circles(
    gray: NDArray[np.uint8],
    *,
    dp: float = 1.2,
    min_dist: int = 50,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 20,
    max_radius: int = 150,
) -> NDArray[np.int32] | None:
    """Run Hough circle detection. Returns ``(N, 3)`` int array of ``(x, y, r)`` or None.

    Wraps ``cv2.HoughCircles`` and returns the rounded result. Notebook cell 29.
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype(np.int32)


# ---------------------------------------------------------------------------
# Border & shape measurement (Stage 10)
# ---------------------------------------------------------------------------


def radial_profile(
    mask: NDArray[np.uint8],
) -> tuple[tuple[float, float], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``((cx, cy), radii, angles)`` for the largest external contour.

    The radii/angles arrays are sorted by angle (``[-π, π]``) so the caller can
    plot them as a continuous profile without re-sorting. Uses
    ``CHAIN_APPROX_NONE`` so every contour pixel contributes a sample (needed
    for a smooth radial curve).

    Returns ``((0.0, 0.0), empty, empty)`` if the mask has no contours or no
    foreground pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (0.0, 0.0), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return (0.0, 0.0), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    pts = contour[:, 0, :].astype(np.float64)  # (N, 2)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    radii = np.sqrt(dx * dx + dy * dy)
    angles = np.arctan2(dy, dx)  # [-π, π]

    order = np.argsort(angles)
    return (cx, cy), radii[order], angles[order]


def asymmetry_score(mask: NDArray[np.uint8]) -> float:
    """Horizontal-flip IoU around the lesion's centroid (Stage 10 metric).

    Pipeline: translate so the centroid lands on the image center, flip
    horizontally, compute the IoU between the centered original and the
    flipped copy. Returns ``1.0`` for a perfectly symmetric lesion, ``0.0``
    for fully disjoint halves.

    Edge cases: empty mask returns ``1.0``; degenerate (1-px) masks return
    ``1.0`` because both halves trivially coincide.
    """
    h, w = mask.shape
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 1.0

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    # Translate centroid → image center. INTER_NEAREST preserves binary mask.
    M = np.float32([[1.0, 0.0, w / 2.0 - cx], [0.0, 1.0, h / 2.0 - cy]])
    centered = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    flipped = cv2.flip(centered, 1)

    a = centered > 0
    b = flipped > 0
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0
    return float(intersection / union)


def border_irregularity(
    contour: NDArray[np.int32],
    centroid: tuple[float, float],
) -> float:
    """Coefficient of variation of the radial distance from ``centroid`` to
    every point on ``contour``.

    Returns the dimensionless ratio ``std(radii) / mean(radii)``. A perfect
    circle scores ``0.0``; values ``> 0.3`` indicate a clearly irregular
    border. Returns ``0.0`` for empty contours or degenerate (single-pixel)
    contours where the mean radius is undefined.
    """
    if contour is None or len(contour) == 0:
        return 0.0

    cx, cy = centroid
    pts = contour[:, 0, :].astype(np.float64)
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    if radii.size == 0:
        return 0.0
    mean_r = float(np.mean(radii))
    if mean_r <= 0:
        return 0.0
    return float(np.std(radii) / mean_r)


def asymmetry_score_2axis(
    mask: NDArray[np.uint8],
) -> tuple[float, float]:
    """Two-axis asymmetry (Stage 13 / ABCD rule of dermoscopy).

    Returns ``(h_asym, v_asym)``, each in ``[0, 1]``:

        h_asym = 1 − IoU(mask, flip_h(mask) aligned at centroid)
        v_asym = 1 − IoU(mask, flip_v(mask) aligned at centroid)

    The ABCD rule sums these into a single 2-axis score::

        A_2 = h_asym + v_asym ∈ [0, 2]

    where ``0`` = symmetric on both axes, ``1`` = asymmetric on one axis,
    ``2`` = asymmetric on both axes.

    Empty or degenerate masks return ``(0.0, 0.0)`` (treated as fully
    symmetric).
    """
    h, w = mask.shape
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return 0.0, 0.0

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    M = np.float32(
        [[1.0, 0.0, w / 2.0 - cx], [0.0, 1.0, h / 2.0 - cy]]
    )
    centered = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    def _iou(flip_axis: int) -> float:
        flipped = cv2.flip(centered, flip_axis)
        a = centered > 0
        b = flipped > 0
        inter = int(np.sum(a & b))
        union = int(np.sum(a | b))
        if union == 0:
            return 1.0
        return float(inter / union)

    iou_h = _iou(1)  # horizontal flip
    iou_v = _iou(0)  # vertical flip

    # IoU=1.0 (perfect symmetry) → asym = 0; IoU=0 (no overlap) → asym = 1.
    h_asym = float(np.clip(1.0 - iou_h, 0.0, 1.0))
    v_asym = float(np.clip(1.0 - iou_v, 0.0, 1.0))
    return h_asym, v_asym


def equivalent_diameter(mask_uint8: NDArray[np.uint8]) -> tuple[float, float]:
    """Equivalent-circular diameter + major-axis length (Stage 12 / ABCD-D).

    Returns ``(d_eq_px, d_major_px)`` in pixels:

        d_eq   = sqrt(4 · area / π)  — diameter of a circle with the same
                                       area as the lesion.
        d_major = major-axis length of ``cv2.fitEllipse`` on the largest
                  external contour. ``0.0`` if the contour has fewer than 5
                  points (OpenCV's minimum for fitEllipse).

    Without a calibration marker in the image the result is in raw pixels.
    The pipeline does not implement mm calibration in Tier 1; the caller
    can convert externally.
    """
    area_px = float(np.sum(mask_uint8 > 0))
    d_eq = float(np.sqrt(max(0.0, 4.0 * area_px / np.pi)))

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    d_major = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            try:
                (_, _), (ea, eb), _ = cv2.fitEllipse(c)
                # fitEllipse returns (width, height); the major axis is the
                # longer of the two.
                d_major = float(max(ea, eb))
            except cv2.error:
                d_major = 0.0
    return d_eq, d_major