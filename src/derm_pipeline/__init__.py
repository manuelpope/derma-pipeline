"""Dermatoscopio image processing pipeline.

Local, reusable replacement for the original `img_filtros_dermatoscopio.ipynb`
Colab notebook. Exposes both a CLI (`derm_pipeline.cli`) and a FastAPI server
(`derm_pipeline.api`) over the same pure-function core.
"""

from derm_pipeline.borders import crop_zooms, find_critical_points
from derm_pipeline.enhancement import enhance_color_lab
from derm_pipeline.io import load_image
from derm_pipeline.metrics import compute_metrics, lesions_to_df
from derm_pipeline.preprocess import preprocess
from derm_pipeline.segmentation import find_lesions

__all__ = [
    "load_image",
    "preprocess",
    "find_lesions",
    "enhance_color_lab",
    "find_critical_points",
    "crop_zooms",
    "lesions_to_df",
    "compute_metrics",
]

__version__ = "0.1.0"