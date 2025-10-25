# -*- coding: utf-8 -*-
"""
mask2poly.py

Convert binary masks of nuclei into polygon coordinate arrays and save them
as MATLAB .mat files.

Each mask is expected to be a 2D binary image (nuclei=1 or 255, background=0).
Contours of individual nuclei are extracted using OpenCV and stored as a list
of NumPy arrays, each containing the (x, y) coordinates of one nucleus.

This process is the inverse of poly2mask, useful for exporting predictions to
MATLAB-compatible datasets.

Parameters
----------
set_data : list[np.ndarray]
    List of binary masks (each as a 2D NumPy array).
data_img_names : list[str]
    List of filenames (without extension) corresponding to each mask.
output_dir : str
    Directory path where the resulting .mat files will be saved.

Output
------
Creates one .mat file per mask, each containing a variable 'contours'
with the list of polygon coordinates.
"""

from __future__ import annotations
from typing import List
import os
import cv2
import numpy as np
from scipy.io import savemat


def mask2poly(
    set_data: List[np.ndarray],
    data_img_names: List[str],
    output_dir: str
) -> None:
    """
    Convert binary masks into polygon coordinate lists and save as .mat files.

    Parameters
    ----------
    set_data : list[np.ndarray]
        Binary masks (2D arrays).
    data_img_names : list[str]
        Base filenames for saving .mat files.
    output_dir : str
        Destination folder for output .mat files.
    """

    os.makedirs(output_dir, exist_ok=True)  # ensure output directory exists

    for idx, mask in enumerate(set_data):
        # Ensure mask is uint8 and strictly binary (0 / 255)
        binary_mask = mask.astype(np.uint8)
        _, binary_mask = cv2.threshold(binary_mask, 0, 255, cv2.THRESH_BINARY)

        # Find contours for each nucleus (external boundaries only)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        contours_cell: list[np.ndarray] = []

        for contour in contours:
            # Convert contour (N,1,2) -> (N,2)
            coords = contour.squeeze(axis=1).astype(np.float32)
            contours_cell.append(coords)

        # Build full path (ensuring proper .mat extension)
        base_name = os.path.splitext(data_img_names[idx])[0]
        mat_path = os.path.join(output_dir, f"{base_name}.mat")

        # Save the contour list as a MATLAB-compatible structure
        savemat(mat_path, {"contours": contours_cell})

        print(f"Saved: {mat_path} ({len(contours_cell)} objects found)")