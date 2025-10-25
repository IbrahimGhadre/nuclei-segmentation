# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import scipy.ndimage as ndimage


def label_instances(
    masks: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Label connected objects in binary masks.

    Each binary prediction mask is converted into an instance map where:
    - background pixels = 0
    - each connected component = 1, 2, 3, ...

    Parameters
    ----------
    masks : list[np.ndarray]
        List of predicted masks (2D arrays). Each mask can be bool, 0/1, or 0/255.

    Returns
    -------
    labeled_list : list[np.ndarray]
        For each input mask, a 2D array of the same shape where each nucleus/object
        has a unique integer ID.
    counts : list[int]
        Number of connected components found in each mask.
    """

    labeled_list: List[np.ndarray] = []
    counts: List[int] = []

    for m in masks:
        # Handle empty masks robustly (all zeros)
        max_val = m.max()
        if max_val == 0:
            # No foreground at all -> empty label map
            labeled = np.zeros_like(m, dtype=int)
            num_features = 0
        else:
            # Normalize to {0,1} and convert to int
            bin_mask = (m / max_val).astype(int)

            # Connected-component labeling
            labeled, num_features = ndimage.label(bin_mask)

        labeled_list.append(labeled)
        counts.append(num_features)

    return labeled_list, counts
