"""Color enhancement in the LAB color space.

Lifted verbatim from notebook cell 21 step 1: CLAHE on the L channel for
local contrast, mild saturation boost on the a/b channels. Used both as the
final visual layer for the critical-point zooms and as the input to the
gradient + std scoring in `borders.find_critical_points`.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def enhance_color_lab(
    rgb: NDArray[np.uint8],
    *,
    clip_limit: float = 2.0,
    grid: tuple[int, int] = (8, 8),
    ab_alpha: float = 1.15,
    ab_beta: int = 0,
) -> NDArray[np.uint8]:
    """Apply a LAB-based high-fidelity color enhancement.

    Args:
        rgb: Input image as ``uint8`` RGB ndarray.
        clip_limit: CLAHE clip limit on the L channel.
        grid: CLAHE tile grid size.
        ab_alpha: Gain multiplier on the a/b (chromatic) channels.
        ab_beta: Bias added after the a/b gain.

    Returns:
        Enhanced ``uint8`` RGB ndarray of the same shape as the input.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_enhanced = clahe.apply(l_channel)

    a_enhanced = cv2.convertScaleAbs(a_channel, alpha=ab_alpha, beta=ab_beta)
    b_enhanced = cv2.convertScaleAbs(b_channel, alpha=ab_alpha, beta=ab_beta)

    merged = cv2.merge((l_enhanced, a_enhanced, b_enhanced))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)