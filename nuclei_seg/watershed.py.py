from __future__ import annotations
from typing import List
import cv2
import numpy as np


def apply_watershed(masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Marker-controlled watershed post-processing for nuclei segmentation.

    This function takes binary segmentation masks (one per image), and
    applies a classical marker-controlled watershed to split touching
    nuclei into separate instances.

    For each input mask:
    1. Compute sure foreground using distance transform.
    2. Compute sure background using dilation.
    3. Mark the unknown (border) region.
    4. Run cv2.watershed() to split touching objects.
    5. Draw watershed boundaries (where markers == -1) as black lines.

    Parameters
    ----------
    masks : list[np.ndarray]
        List of 2D binary masks (uint8 or bool-like), where nuclei pixels
        are non-zero and background is 0.

    Returns
    -------
    processed_masks : list[np.ndarray]
        List of 2D uint8 masks after watershed refinement. The output mask
        has nuclei regions and explicit black boundaries separating adjacent
        nuclei. (Note: this is not yet an "instance ID map"; it's still a
        grayscale mask with cut lines.)
    """

    processed_masks: List[np.ndarray] = []

    for m in masks:
        # Ensure uint8 single-channel (0 background, >0 foreground)
        img = m.astype(np.uint8)

        # Edge case: completely empty mask -> just append zeros and continue
        if img.max() == 0:
            processed_masks.append(np.zeros_like(img, dtype=np.uint8))
            continue

        # Watershed in OpenCV needs a 3-channel image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Morphological dilation to get "sure background"
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(img, kernel, iterations=10)

        # Distance transform to get "sure foreground"
        dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        _, sure_fg = cv2.threshold(
            dist_transform,
            0.7 * dist_transform.max(),
            255,
            0
        )
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown region = background minus foreground seeds
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Connected components on the foreground seeds -> markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 10  # shift to avoid 0/1 ambiguity

        # Mark unknown pixels with 0, so watershed will assign them
        markers[unknown == 255] = 0

        # Apply watershed. After this:
        # - markers == -1 are watershed boundaries
        # - markers > 1 are different catchment basins
        markers = cv2.watershed(img_bgr, markers)

        # Visualize / convert to mask with explicit boundaries
        # We'll paint boundaries black (0), everything else white-ish.
        # Start from a copy of the original nucleus mask:
        post_mask_bgr = img_bgr.copy()
        post_mask_bgr[markers == -1] = [0, 0, 0]

        # Save only one channel (grayscale-like)
        processed_masks.append(post_mask_bgr[:, :, 0])

    return processed_masks
