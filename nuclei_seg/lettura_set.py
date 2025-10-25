# -*- coding: utf-8 -*-
"""
lettura_set.py - dataset loading utilities for nuclei segmentation (colon H&E).

- poly2mask(vertices, shape): rasterizes polygon vertices into a binary mask
  where borders are excluded (edge pixels set to False).
- lettura_set(...): builds:
    * dict_data: {'image': [RGB images], 'segmentation': [boolean masks]}
    * N_gt_list: list of nuclei counts per image
"""

from __future__ import annotations
from typing import List, Tuple, Union
from pathlib import Path
import os
import logging

import cv2
import numpy as np
from scipy.io import loadmat
from skimage import draw

logger = logging.getLogger(__name__)

__all__ = ["poly2mask", "lettura_set"]


def poly2mask(
    vertex_row_coords: np.ndarray,
    vertex_col_coords: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterize a polygon into a boolean mask with borders removed."""
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape
    )
    mask = np.zeros(shape, dtype=bool)
    mask[fill_row_coords, fill_col_coords] = True   # fill polygon
    mask[vertex_row_coords, vertex_col_coords] = False  # remove edges
    return mask


def lettura_set(
    data_img: List[str],
    data_img_path: Union[str, Path],
    data: str,
    base_dataset_dir: Union[str, Path] = Path("/content/DATASET"),
):
    """
    Build dataset dict and nuclei counts list from image names and .mat annotations.

    Parameters
    ----------
    data_img : list[str]
        Filenames of images in the set (train/val/test).
    data_img_path : str | Path
        Directory path containing the images.
    data : str
        Set name: "train", "val", or "test".
    base_dataset_dir : str | Path, optional
        Root folder of the dataset (defaults to '/content/DATASET').
        Ground-truth .mat files are expected under:
        <base_dataset_dir>/<data>/manual/<image_name>.mat

    Returns
    -------
    dict_data : dict
        {'image': [RGB np.ndarray], 'segmentation': [bool masks with borders removed]}
    N_gt_list : list[int]
        Number of nuclei per image.
    """
    data_img_path = Path(data_img_path)
    base_dataset_dir = Path(base_dataset_dir)

    dict_data = {"image": [], "segmentation": []}
    N_gt_list = []

    for name in data_img:
        # Read image
        img_path = data_img_path / name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dict_data["image"].append(img)
        h, w = img.shape[:2]

        # Read corresponding manual annotations (.mat)
        mat_path = base_dataset_dir / data / "manual" / f"{name[:-4]}.mat"
        xy = loadmat(str(mat_path))          # cell array with one polygon per nucleus
        xyy = xy["xy"]

        manual_mask = np.zeros((h, w), dtype=bool)
        N_gt = xyy.shape[1]
        N_gt_list.append(N_gt)

        for i in range(N_gt):
            current_int = xyy[0, i] - 1      # MATLAB 1-based -> Python 0-based
            mask_temp = poly2mask(
                current_int[:, 1],           # rows (y)
                current_int[:, 0],           # cols (x)
                (h, w),
            )
            manual_mask |= mask_temp

        dict_data["segmentation"].append(manual_mask)

    logger.debug("Built dataset with %d images for split '%s'", len(data_img), data)
    return dict_data, N_gt_list