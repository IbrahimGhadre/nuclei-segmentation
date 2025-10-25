# -*- coding: utf-8 -*-
"""
divisione_in_patch.py - extract overlapping patches (image + mask) and save them to disk.

This function slides a fixed-size square window over each image/mask pair and
creates overlapping patches. Each patch is then saved to disk with a filename
that encodes:
    - i  -> index of the source image in the dataset (1-based)
    - k  -> row index of the patch within that image (1-based)
    - j  -> column index of the patch within that image (1-based)

Patch extraction details
------------------------
- Patches from RGB images are extracted with patch size (shape, shape, 3).
- Patches from binary masks are extracted with patch size (shape, shape).
- The sliding step is computed as: step = int(overlap * shape)

  NOTE:
  Here "overlap" behaves like a stride fraction. For example:
      shape = 256
      overlap = 0.5  -> step = 128  (≈50% overlap between neighboring patches)
      overlap = 1.0  -> step = 256  (no overlap: patches just tile the image)

Inputs
------
data_img : list[str]
    Filenames of the original images (used for indexing/traceability).
dict_data : dict
    Dictionary with:
        'image'        -> list of RGB images as np.ndarray [H, W, 3]
        'segmentation' -> list of boolean masks as np.ndarray [H, W]
path_img : str or pathlib.Path
    Directory where extracted image patches will be saved.
path_mask : str or pathlib.Path
    Directory where extracted mask patches will be saved.
shape : int
    Patch size (square side length).
overlap : float
    Fraction that determines the patch step via step = int(overlap * shape).

Outputs
-------
dizionario_patches : dict
    Dictionary with:
        'nrows' -> list of last patch row index for each input image
        'ncols' -> list of last patch col index for each input image
    NOTE: we intentionally keep this behavior for compatibility with downstream code.
"""

from __future__ import annotations
from typing import Dict, List, Union
from pathlib import Path

import numpy as np
import cv2
from patchify import patchify


__all__ = ["divisione_in_patch"]


def divisione_in_patch(
    data_img: List[str],
    dict_data: Dict[str, List[np.ndarray]],
    path_img: Union[str, Path],
    path_mask: Union[str, Path],
    shape: int,
    overlap: float,
) -> Dict[str, List[int]]:
    # Ensure output dirs are Path objects and exist
    path_img = Path(path_img)
    path_mask = Path(path_mask)
    path_img.mkdir(parents=True, exist_ok=True)
    path_mask.mkdir(parents=True, exist_ok=True)

    dizionario_patches = {"nrows": [], "ncols": []}

    # iterate over each image in the dataset
    for i in range(len(data_img)):
        # Extract all patches for the RGB image
        patches_img = patchify(
            dict_data["image"][i],
            (shape, shape, 3),
            step=int(overlap * shape),
        )

        # Extract all patches for the binary mask
        patches_mask = patchify(
            dict_data["segmentation"][i],
            (shape, shape),
            step=int(overlap * shape),
        )

        # Loop over spatial grid of patches
        for k in range(patches_img.shape[0]):        # patch row index
            for j in range(patches_img.shape[1]):    # patch col index
                # Extract the (k, j)-th RGB patch.
                # patches_img[k, j, ...] has shape (1, shape, shape, 3)
                single_patch = np.squeeze(
                    patches_img[k, j, :, :, :, :],
                    axis=0
                )  # -> (shape, shape, 3)

                # Extract the corresponding mask patch.
                # patches_mask[k, j, ...] has shape (shape, shape)
                single_patch_mask = patches_mask[k, j, :, :]

                # Convert boolean mask to 0/255 uint8 for saving as JPEG
                mask_8bit = (single_patch_mask.astype(np.uint8)) * 255

                # Save image patch
                img_filename = f"immagine_{i+1}_{k+1}_{j+1}.jpg"
                cv2.imwrite(
                    str(path_img / img_filename),
                    single_patch,
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                )

                # Save mask patch (also JPEG, same naming style as original code)
                mask_filename = f"mask_{i+1}_{k+1}_{j+1}.jpg"
                cv2.imwrite(
                    str(path_mask / mask_filename),
                    mask_8bit,
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                )

        # Record patch grid info for this image
        # NOTE: this keeps the original semantics:
        # append the last indices k and j seen in the loops above.
        dizionario_patches["nrows"].append(k)
        dizionario_patches["ncols"].append(j)

    return dizionario_patches