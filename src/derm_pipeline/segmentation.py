"""Lesion segmentation from a connected-component label map.

Replaces the inline `for i in range(1, num_labels): ... MIN_AREA = 5000` loop
from notebook cell 13. Pure function — no shared state, no plotting.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def find_lesions(
    labels: NDArray[np.int32],
    num_labels: int,
    image_shape: tuple[int, int],
    *,
    stats: NDArray[np.int32] | None = None,
    centroids: NDArray[np.float64] | None = None,
    min_area: int = 5000,
    max_image_fraction: float = 0.8,
) -> list[dict]:
    """Convert a CC label map into a list of lesion descriptors.

    Mirrors cell 13's filter: drop components below ``min_area`` and the
    background-sized component that spans more than ``max_image_fraction`` of
    the image.

    Args:
        labels: ``int32`` label map from ``cv2.connectedComponentsWithStats``.
            Kept in the signature so callers can pass the same map; not used
            when ``stats``/``centroids`` are provided (the recommended path).
        num_labels: Number of labels in the map (including background 0).
        image_shape: ``(H, W)`` of the source image, used for the fraction cap.
        stats: Pre-computed CC ``stats`` matrix from
            ``cv2.connectedComponentsWithStats``. If ``None``, ``centroids``
            must also be ``None`` and the function will recompute CC on
            ``labels`` (only valid if ``labels`` is a ``uint8`` binary mask).
        centroids: Pre-computed CC ``centroids`` matrix; see ``stats``.
        min_area: Minimum component area in pixels to keep.
        max_image_fraction: Components covering more than this fraction of the
            image are treated as background and dropped.

    Returns:
        List of dicts sorted by area (descending)::

            {
                "id":     int,    # label id from cv2
                "x":      int,    # bbox left
                "y":      int,    # bbox top
                "width":  int,
                "height": int,
                "area":   int,    # pixel count (CC stat)
                "cx":     float,  # centroid x
                "cy":     float,  # centroid y
            }
    """
    h, w = image_shape[:2]
    total_pixels = float(h * w)

    if stats is None or centroids is None:
        # Fallback path: recompute CC. Requires a uint8 binary mask, not the
        # int32 label map produced by `preprocess`.
        if (stats is None) != (centroids is None):
            raise ValueError("stats and centroids must both be supplied or both omitted")
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(labels)

    lesions: list[dict] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        # background-sized cap
        if area > total_pixels * max_image_fraction:
            continue
        if area < min_area:
            continue

        lesions.append(
            {
                "id": int(i),
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "width": int(stats[i, cv2.CC_STAT_WIDTH]),
                "height": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": area,
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
            }
        )

    lesions.sort(key=lambda d: d["area"], reverse=True)
    return lesions


def lesion_mask(labels: NDArray[np.int32], lesion_id: int) -> NDArray[np.uint8]:
    """Build a binary mask (0/255) for a single lesion id."""
    return np.uint8(labels == lesion_id) * 255


def lesion_contours(
    mask: NDArray[np.uint8],
) -> tuple[NDArray[np.int32] | None, list]:
    """Return ``(biggest_contour, all_contours)`` for a binary lesion mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, []
    biggest = max(contours, key=cv2.contourArea)
    return biggest, contours