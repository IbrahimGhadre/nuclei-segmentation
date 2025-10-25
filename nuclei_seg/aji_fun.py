# -*- coding: utf-8 -*-
"""
aji_fun.py

Compute the Aggregated Jaccard Index (AJI) for instance segmentation of nuclei.

Given:
- a manual annotation (ground truth) stored in MATLAB `.mat` format, where each
  nucleus is represented as a polygon,
- a predicted segmentation map where each nucleus instance is labeled with a
  unique integer ID (background = 0),

this function matches each ground-truth nucleus to at most one predicted
nucleus (the one with highest Jaccard overlap), accumulates pixel-wise
intersections and unions, and then computes the global AJI score.

References:
AJI = sum_i |G_i ∩ P*(i)| / ( sum_i |G_i ∪ P*(i)|  +  sum_{k∈unmatched P} |P_k| )
where P*(i) is the best-matching prediction for ground-truth instance G_i.

Outputs:
- a single float in [0, 1].
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from skimage import draw


def _poly2mask_no_border(
    vertex_row_coords: np.ndarray,
    vertex_col_coords: np.ndarray,
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert a polygon (list of vertex coordinates) to a binary mask.

    Parameters
    ----------
    vertex_row_coords : np.ndarray
        Row (y) coordinates of the polygon vertices.
    vertex_col_coords : np.ndarray
        Column (x) coordinates of the polygon vertices.
    shape : (int, int)
        Target mask shape (height, width).

    Returns
    -------
    mask : np.ndarray, dtype=uint8
        Binary mask of shape `shape`, where pixels inside the polygon are 1,
        background is 0.
    """
    fill_row, fill_col = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row, fill_col] = 1
    return mask


def aji_fun(
    gt_mat_path: str,
    predicted_map: np.ndarray
) -> float:
    """
    Compute the Aggregated Jaccard Index (AJI) between ground-truth nuclei
    and predicted nuclei.

    Parameters
    ----------
    gt_mat_path : str
        Path to the .mat file containing manual annotations.
        The .mat file is expected to have a variable 'xy', where:
        - xy[0, i] is an (N_i x 2) array of [x,y] polygon vertices
          for the i-th ground-truth nucleus.
        - Coordinates are MATLAB-style (1-based).
    predicted_map : np.ndarray
        2D array where each predicted nucleus instance has a unique integer ID
        (1, 2, 3, ...). Background is 0. Shape must match the target image size.

    Returns
    -------
    aji : float
        Aggregated Jaccard Index in [0, 1].
    """

    # ---------------------------------------------------------------------
    # Load ground-truth nuclei polygons from the .mat file
    # ---------------------------------------------------------------------
    mat_data = loadmat(gt_mat_path)
    xy_polygons = mat_data["xy"]        # shape: (1, N_gt), each cell is an array of polygon coords
    num_gt = xy_polygons.shape[1]       # number of ground-truth nuclei

    # ---------------------------------------------------------------------
    # Collect predicted nuclei instance IDs (exclude background = 0)
    # We'll also track which predicted nuclei have been "matched"
    # ---------------------------------------------------------------------
    unique_pred_ids = np.unique(predicted_map)
    unique_pred_ids = unique_pred_ids[unique_pred_ids != 0]  # drop background

    # pr_list is a 2-column array:
    #   col 0 -> nucleus ID in predicted_map
    #   col 1 -> how many times we've matched that predicted nucleus to a GT one
    pr_list = np.column_stack(
        (unique_pred_ids, np.zeros((unique_pred_ids.shape[0], 1), dtype=int))
    )

    # Running totals for AJI numerator (intersection) and denominator (union)
    total_intersection = 0
    total_union = 0

    # ---------------------------------------------------------------------
    # Loop over each ground-truth nucleus
    # ---------------------------------------------------------------------
    for i in range(num_gt):
        # xy_polygons[0, i] is an array of shape (Ni, 2) with [x, y] coordinates in MATLAB convention
        # We subtract 1 to convert from MATLAB 1-based to Python 0-based indices.
        gt_coords = xy_polygons[0, i] - 1

        # Build a binary mask for this ground-truth nucleus
        # gt_mask has shape == predicted_map.shape and values in {0,1}
        gt_mask = _poly2mask_no_border(
            vertex_row_coords=gt_coords[:, 1],
            vertex_col_coords=gt_coords[:, 0],
            shape=predicted_map.shape
        )

        # Multiply gt_mask * predicted_map to see which predicted labels overlap this GT nucleus.
        # Wherever gt_mask==1, predicted_map retains its label; elsewhere 0.
        overlap_label_map = gt_mask * predicted_map

        if np.count_nonzero(overlap_label_map) == 0:
            # No predicted nucleus overlaps this GT nucleus at all.
            # That means it's a complete miss (false negative),
            # and its area counts only in the denominator (union).
            total_union += np.count_nonzero(gt_mask)
            continue

        # Otherwise: there is at least one predicted nucleus overlapping this GT nucleus.
        # Extract which predicted instance IDs are overlapping (exclude 0/background).
        overlapping_pred_ids = np.unique(overlap_label_map)
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids != 0]

        # We now select the *best matching* predicted nucleus for this GT nucleus,
        # based on the Jaccard Index (IoU = intersection/union).
        best_jaccard = 0.0
        best_match_id = None

        for pred_id in overlapping_pred_ids:
            # Binary mask of the current predicted nucleus
            pred_mask = (predicted_map == pred_id).astype(np.uint8)

            intersection = np.count_nonzero(np.logical_and(gt_mask, pred_mask))
            union = np.count_nonzero(np.logical_or(gt_mask, pred_mask))

            if union == 0:
                # Should not really happen, but guard division by zero
                continue

            jaccard = intersection / union  # IoU / JI

            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match_id = pred_id

        # Build a mask of ONLY the best-matching predicted nucleus
        best_pred_mask = (predicted_map == best_match_id).astype(np.uint8)

        # Update total intersection and total union
        total_intersection += np.count_nonzero(np.logical_and(gt_mask, best_pred_mask))
        total_union += np.count_nonzero(np.logical_or(gt_mask, best_pred_mask))

        # Mark this predicted nucleus as "used"
        # Find its row in pr_list (column 0 is the predicted ID)
        match_row = np.where(pr_list[:, 0] == best_match_id)[0][0]
        pr_list[match_row, 1] += 1

    # ---------------------------------------------------------------------
    # Handle unmatched predicted nuclei:
    # Any predicted nucleus that was never assigned to any GT nucleus
    # contributes ONLY to the denominator.
    # ---------------------------------------------------------------------
    unused_mask_rows = np.where(pr_list[:, 1] == 0)[0]

    for row_idx in unused_mask_rows:
        pred_id = pr_list[row_idx, 0]
        pred_mask = (predicted_map == pred_id).astype(np.uint8)
        total_union += np.count_nonzero(pred_mask)

    # ---------------------------------------------------------------------
    # Final AJI
    # ---------------------------------------------------------------------
    if total_union == 0:
        # Edge case: no nuclei at all
        return 0.0

    aji = total_intersection / total_union
    return float(aji)