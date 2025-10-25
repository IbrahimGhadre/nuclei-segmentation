# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple
import os

import cv2
import numpy as np
from scipy.io import loadmat
from skimage import draw


def poly2mask_with_borders(
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Rasterize a polygon into a boolean mask, keeping the full nucleus
    including its border.

    Parameters
    ----------
    row_coords : np.ndarray
        Y-coordinates (row indices) of the nucleus contour.
    col_coords : np.ndarray
        X-coordinates (column indices) of the nucleus contour.
    shape : (H, W)
        Output mask shape.

    Returns
    -------
    mask : np.ndarray of bool, shape (H, W)
        True inside the nucleus polygon (borders included),
        False elsewhere.
    """
    rr, cc = draw.polygon(row_coords, col_coords, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def load_dataset_with_borders(
    image_names: List[str],
    image_dir: str,
    annotations_root: str,
    subset_name: str,
) -> Tuple[Dict[str, List[np.ndarray]], List[int]]:
    """
    Load RGB images and their manual nucleus segmentations (borders included).

    Each annotation is stored in a MATLAB .mat file with a variable 'xy',
    where 'xy' is a cell array (size 1 x N_gt). Each cell contains the
    contour coordinates of one nucleus as an (M_i x 2) array:
    [x_coords, y_coords] in MATLAB indexing (1-based).

    We convert each contour to a boolean mask using poly2mask_with_borders(),
    and OR all nuclei together into a single binary mask.

    Parameters
    ----------
    image_names : list[str]
        Filenames of the histology images (e.g. ["img_01.png", ...]).
    image_dir : str
        Directory containing those RGB images.
    annotations_root : str
        Root directory where annotations live.
        Expected structure:
            {annotations_root}/{subset_name}/manual/<image>.mat
    subset_name : str
        Name of the subset, e.g. "train", "val", "test".
        Used to pick the right subfolder under annotations_root.

    Returns
    -------
    dataset : dict
        {
            "image": [np.ndarray(H,W,3) RGB],
            "segmentation": [np.ndarray(H,W) bool],  # nuclei mask
        }
    nuclei_counts : list[int]
        nuclei_counts[k] = number of annotated nuclei in image k.
    """
    dataset = {
        "image": [],
        "segmentation": [],
    }
    nuclei_counts: List[int] = []

    for idx, fname in enumerate(image_names):
        # --- Load RGB image
        bgr = cv2.imread(os.path.join(image_dir, fname))
        if bgr is None:
            raise FileNotFoundError(
                f"Could not read image '{fname}' in '{image_dir}'"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dataset["image"].append(rgb)

        H, W = rgb.shape[:2]

        # --- Load corresponding .mat annotation file
        # replace extension (.png/.jpg/...) with .mat
        base_name, _ext = os.path.splitext(fname)
        mat_path = os.path.join(
            annotations_root,
            subset_name,
            "manual",
            base_name + ".mat",
        )

        mat_data = loadmat(mat_path)
        # mat_data["xy"] is 1 x N_gt cell-like
        xy_cells = mat_data["xy"]

        # Prepare an empty mask for this image
        mask_bool = np.zeros((H, W), dtype=bool)

        # N_gt = number of annotated nuclei
        N_gt = xy_cells.shape[1]
        nuclei_counts.append(N_gt)

        for n in range(N_gt):
            # MATLAB coordinates are 1-based; convert to 0-based
            contour_coords = xy_cells[0, n] - 1
            # contour_coords shape: (M, 2) = [[x0,y0],[x1,y1],...]

            # Note: first column = x (col), second column = y (row)
            x_coords = contour_coords[:, 0]
            y_coords = contour_coords[:, 1]

            nucleus_mask = poly2mask_with_borders(
                row_coords=y_coords,
                col_coords=x_coords,
                shape=(H, W),
            )
            # Combine nucleus into global mask
            mask_bool |= nucleus_mask

        dataset["segmentation"].append(mask_bool)

    return dataset, nuclei_counts
