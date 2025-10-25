# -*- coding: utf-8 -*-
"""
unpatchify_and_unpadding.py

This function reconstructs full-size segmentation masks from predicted patches,
removing the mirror padding added during preprocessing. Each patch is center-cropped
before recomposition, and the final image is trimmed back to the original size.

Inputs:
- data_img: list of image filenames
- path_mask_post_rete: path to folder containing predicted masks from the model
                       (files named "immagine_i_j_k.png")
- padding: dictionary with padding information for each image (top, bottom, left, right)
- dizionario_patches: dictionary containing 'nrows' and 'ncols' (patch grid layout)
- patch_size_crop: size of the center region extracted from each patch

Output:
- maschere_unpatch: list of reconstructed masks with original dimensions
"""

from __future__ import annotations
import numpy as np
import cv2
import os
from typing import List, Dict

def unpatchify_and_unpadding(
    data_img: List[str],
    path_mask_post_rete: str,
    padding: Dict[str, List[int]],
    dizionario_patches: Dict[str, List[int]],
    patch_size_crop: int
) -> List[np.ndarray]:
    """
    Rebuilds full-size masks from predicted patches and removes mirror padding.

    Parameters
    ----------
    data_img : List[str]
        List of original image filenames.
    path_mask_post_rete : str
        Directory containing predicted patch masks (named as "immagine_i_j_k.png").
    padding : Dict[str, List[int]]
        Dictionary with padding values for each image (top, bottom, left, right).
    dizionario_patches : Dict[str, List[int]]
        Dictionary with the number of patches per image row/column.
    patch_size_crop : int
        Size of the cropped central area extracted from each patch.

    Returns
    -------
    maschere_unpatch : List[np.ndarray]
        List of reconstructed masks (same size as original images).
    """
    
    mask_files = os.listdir(path_mask_post_rete)
    maschere_unpatch: List[np.ndarray] = []

    patch_size = np.array([patch_size_crop, patch_size_crop])

    for k, _ in enumerate(data_img):
        num_rows = dizionario_patches['nrows'][k] + 1
        num_cols = dizionario_patches['ncols'][k] + 1

        # Initialize empty reconstructed mask
        reconstructed_image = np.zeros(
            (num_rows * patch_size[0], num_cols * patch_size[1]),
            dtype=np.float32
        )

        # Loop over patch grid
        for i in range(num_rows):
            for j in range(num_cols):
                filename = f"immagine_{k+1}_{i+1}_{j+1}.png"
                mask_path = os.path.join(path_mask_post_rete, filename)
                maschera_pred = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if maschera_pred is None:
                    # Hard fail: we expected this patch and it's missing.
                    # We stop immediately so the user knows something is wrong.
                    raise FileNotFoundError(
                        f"Missing predicted patch '{filename}' for image index {k} "
                        f"(row {i}, col {j}). Expected at: {mask_path}"
                      )  # skip if missing patch

                # Center crop the predicted patch
                h, w = maschera_pred.shape
                start_h = int((h - patch_size_crop) / 2)
                end_h = start_h + patch_size_crop
                start_w = int((w - patch_size_crop) / 2)
                end_w = start_w + patch_size_crop
                cropped_patch = maschera_pred[start_h:end_h, start_w:end_w]

                # Place cropped patch into reconstructed mosaic
                row_start = i * patch_size[0]
                row_end = row_start + patch_size[0]
                col_start = j * patch_size[1]
                col_end = col_start + patch_size[1]

                reconstructed_image[row_start:row_end, col_start:col_end] = cropped_patch

        # Compute unpadding coordinates (remove mirror padding)
        start_x = padding['left'][k] - int(patch_size_crop / 2)
        start_y = padding['top'][k] - int(patch_size_crop / 2)
        end_x = reconstructed_image.shape[1] - padding['right'][k] + int(patch_size_crop / 2)
        end_y = reconstructed_image.shape[0] - padding['bottom'][k] + int(patch_size_crop / 2)

        # Clip indices to avoid out-of-range values
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(reconstructed_image.shape[1], end_x)
        end_y = min(reconstructed_image.shape[0], end_y)

        # Crop the reconstructed image back to original size
        cropped_img = reconstructed_image[start_y:end_y, start_x:end_x]
        maschere_unpatch.append(cropped_img)

    return maschere_unpatch
