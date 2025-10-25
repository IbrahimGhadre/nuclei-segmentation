# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple

import os
import numpy as np
import cv2
import skimage
import skimage.img_as_float  # for static analyzers / clarity


def get_stain_matrix(
    image_rgb: np.ndarray,
    beta: float = 0.15,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Estimate the stain (Hematoxylin / Eosin) color basis using Macenko's method.

    This follows the standard Macenko approach:
    1. Convert RGB to optical density (OD) space.
    2. Keep only sufficiently stained pixels (OD > beta in all channels).
    3. Perform PCA (eigen-decomposition of the OD covariance).
    4. Find the extreme stain directions in the 2D projected OD space.
    5. Return a 2x3 stain matrix W, where each row is a normalized stain vector.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image as uint8 (H, W, 3).
    beta : float
        OD threshold to filter out near-transparent / background pixels.
    alpha : float
        Percentile used to select extreme stain angles (robust min/max).

    Returns
    -------
    W : np.ndarray
        (2, 3) array. Row 0 ~ Hematoxylin stain vector, row 1 ~ Eosin.
        Each row is L2-normalized.
    """
    nrow, ncol, nchan = image_rgb.shape
    img_float = skimage.img_as_float(image_rgb)  # -> float64 in [0,1]

    # Flatten image to (num_pixels, 3)
    pixels = np.zeros((nrow * ncol, nchan))
    for c in range(nchan):
        pixels[:, c] = img_float[:, :, c].reshape(nrow * ncol)

    # Optical Density
    OD = -np.log10(pixels + 1e-8)

    # Keep only "well stained" pixels (remove background / very bright)
    mask_stained = (OD[:, 0] > beta) & (OD[:, 1] > beta) & (OD[:, 2] > beta)
    ODhat = OD[mask_stained]

    # PCA in OD space
    _, V = np.linalg.eig(np.cov(ODhat.T))  # V: eigenvectors as columns
    V = V[:, [1, 0]]  # reorder components (Macenko convention)

    # Project OD data on the first 2 PCs
    Theta = ODhat @ V  # shape: (N, 2)
    Phi = np.arctan2(Theta[:, 1], Theta[:, 0])

    # Robust extremes of the angle distribution
    min_phi = np.percentile(Phi, alpha)
    max_phi = np.percentile(Phi, 100 - alpha)

    # Map back extreme angles into OD space to get stain direction vectors
    vec1 = V @ np.array([[np.cos(min_phi)], [np.sin(min_phi)]])  # (3,1)
    vec2 = V @ np.array([[np.cos(max_phi)], [np.sin(max_phi)]])  # (3,1)

    # Force consistent ordering:
    # Row 0 -> Hematoxylin-like, Row 1 -> Eosin-like
    if vec1[0, 0] > vec2[0, 0]:
        M = np.concatenate((vec1, vec2), axis=1).T  # (2,3)
    else:
        M = np.concatenate((vec2, vec1), axis=1).T  # (2,3)

    # Normalize rows to unit norm
    W = M / np.sqrt(np.sum(M**2, axis=1, keepdims=True))
    return W


def color_deconvolution(
    image_rgb: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform color deconvolution given a stain matrix W.
    Reconstruct pseudo-RGB images for Hematoxylin and Eosin separately.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image as uint8 (H, W, 3).
    W : np.ndarray
        Stain matrix from get_stain_matrix(), shape (2, 3).

    Returns
    -------
    stain_hematoxylin : np.ndarray
        RGB image (uint8) representing only the hematoxylin contribution.
    stain_eosin : np.ndarray
        RGB image (uint8) representing only the eosin contribution.
    """
    nrow, ncol, nchan = image_rgb.shape
    nstains = W.shape[0]  # expected 2

    img_float = skimage.img_as_float(image_rgb)

    # Flatten to (num_pixels, 3)
    pixels = np.zeros((nrow * ncol, nchan))
    for c in range(nchan):
        pixels[:, c] = img_float[:, :, c].reshape(nrow * ncol)

    # Convert to optical density
    OD = -np.log10(pixels + 1e-8)

    # Solve OD ≈ H * W  ->  H ≈ OD * pinv(W)
    # Here we compute H using least-squares via (W W^T)^-1 W.
    H = (np.linalg.solve(W @ W.T, W) @ OD.T).T  # shape: (num_pixels, 2)
    H[H < 0] = 0  # no negative concentrations

    # Reconstruct "pure" stain images by isolating each stain
    # Stain 0 (hematoxylin)
    V_h = (H[:, 0:1] @ W[0:1, :])  # (num_pixels, 3)
    stain_hematoxylin = np.round(
        255 * (10 ** (-V_h))
    ).reshape(nrow, ncol, nchan).astype(np.uint8)

    # Stain 1 (eosin)
    V_e = (H[:, 1:2] @ W[1:2, :])  # (num_pixels, 3)
    stain_eosin = np.round(
        255 * (10 ** (-V_e))
    ).reshape(nrow, ncol, nchan).astype(np.uint8)

    return stain_hematoxylin, stain_eosin


def macenko_decompose_dataset(
    image_names: List[str],
    image_dir: str,
) -> Dict[str, List[np.ndarray]]:
    """
    Run Macenko stain separation on a list of images from a dataset folder.

    For each image:
    - load RGB
    - estimate stain basis (H&E)
    - deconvolve into hematoxylin-only and eosin-only "pseudo-images"

    Parameters
    ----------
    image_names : list[str]
        Filenames of the images in the dataset.
    image_dir : str
        Directory containing those images.

    Returns
    -------
    result : dict
        {
            "name": [...],         # image filename (string)
            "img": [...],          # original RGB image (np.ndarray)
            "stain_Hem": [...],    # hematoxylin RGB component
            "stain_Eos": [...],    # eosin RGB component
        }
    """
    result = {
        "name": [],
        "img": [],
        "stain_Hem": [],
        "stain_Eos": [],
    }

    for fname in image_names:
        # Load image (OpenCV loads as BGR)
        bgr = cv2.imread(os.path.join(image_dir, fname))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {fname} in {image_dir}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Estimate stain matrix and perform color deconvolution
        W = get_stain_matrix(rgb)
        stain_hem, stain_eos = color_deconvolution(rgb, W)

        # Store results
        result["name"].append(fname)
        result["img"].append(rgb)
        result["stain_Hem"].append(stain_hem)
        result["stain_Eos"].append(stain_eos)

    return result
