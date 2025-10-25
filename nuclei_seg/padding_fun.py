# -*- coding: utf-8 -*-
"""
padding_fun.py - mirror padding to make images/masks patch-friendly.

This function applies a two-stage padding to each image/mask pair:
1) Make height and width multiples of `shape` (square patch size) using mirror padding.
2) Add an additional fixed mirror padding of `shape/4` on all sides (top/bottom/left/right)
   to mitigate border artifacts when tiling and recomposing.

Inputs
------
shape : int
    Target square patch size (e.g., 256 or 512).
dict_data : dict
    Dictionary with two keys:
      - 'image': list[np.ndarray H×W×3 in RGB]
      - 'segmentation': list[np.ndarray H×W bool]
data_img : list[str]
    Filenames of the images in the set (its length matches dict_data lists).

Outputs
-------
dict_data : dict
    Same structure as input, but images and masks are mirror-padded.
padding : dict
    Per-image padding information with keys:
      'rows', 'cols'  -> original dimensions
      'top','bottom','left','right' -> applied padding in pixels
      'shape' -> final image shape after padding (H, W, C)
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import logging

import cv2
import numpy as np
import skimage

logger = logging.getLogger(__name__)

__all__ = ["padding_fun"]

def padding_fun(shape: int, dict_data: Dict, data_img: List[str]):
    """
    Apply mirror padding so that each image/mask becomes:
    - divisible by `shape` on both H and W,
    - plus an extra context padding of `shape/4` on all sides.

    Parameters
    ----------
    shape : int
        Square patch size used downstream (and to define extra context).
    dict_data : dict
        {'image': [RGB H×W×3], 'segmentation': [H×W bool]}
    data_img : list[str]
        Filenames corresponding to the images.

    Returns
    -------
    dict_data : dict
        Padded images/masks.
    padding : dict
        Per-sample padding info.
    """
    padding = {
        "rows": [],
        "cols": [],
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
        "shape": [],
    }

    extra = int(shape / 4)  # fixed extra context on each side

    for i in range(len(data_img)):
        # --- Image ---
        img = dict_data["image"][i]
        rows, cols = img.shape[:2]
        padding["rows"].append(rows)
        padding["cols"].append(cols)

        # Compute vertical padding to reach the nearest multiple of `shape`
        if rows % shape != 0:
            need_rows = shape - (rows % shape)
            top = need_rows // 2
            bottom = need_rows - top
        else:
            top = 0
            bottom = 0

        # Compute horizontal padding to reach the nearest multiple of `shape`
        if cols % shape != 0:
            need_cols = shape - (cols % shape)
            left = need_cols // 2
            right = need_cols - left
        else:
            left = 0
            right = 0

        # Add fixed extra context (shape/4) on all sides
        top += extra
        bottom += extra
        left += extra
        right += extra

        padding["top"].append(top)
        padding["bottom"].append(bottom)
        padding["left"].append(left)
        padding["right"].append(right)

        # Mirror padding for the image (RGB)
        padded_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=cv2.BORDER_REFLECT
        )
        dict_data["image"][i] = padded_img
        padding["shape"].append(padded_img.shape)

        # --- Mask ---
        # Convert bool mask -> uint8 (0/255) for OpenCV padding
        mask_u8 = (dict_data["segmentation"][i].astype(np.uint8)) * 255

        padded_mask_u8 = cv2.copyMakeBorder(
            mask_u8, top, bottom, left, right, borderType=cv2.BORDER_REFLECT
        )

        # Back to boolean mask (non-zero -> True)
        padded_mask_bool = skimage.img_as_bool(padded_mask_u8)
        dict_data["segmentation"][i] = padded_mask_bool

    logger.debug("Applied mirror padding (shape=%d, extra=%d) to %d samples.", shape, extra, len(data_img))
    return dict_data, padding