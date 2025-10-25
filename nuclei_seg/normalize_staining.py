from __future__ import annotations
from typing import Dict, List
import cv2
import numpy as np
import staintools


def normalize_staining(
    dataset: Dict[str, List[np.ndarray]],
    image_names: List[str],
    reference_image: np.ndarray,
    method: str = "macenko"
) -> Dict[str, List[np.ndarray]]:
    """
    Normalize H&E staining of histopathology images using the Macenko or
    Vahadane method from `staintools`.

    Parameters
    ----------
    dataset : dict
        Dictionary with keys:
            - "image": list of RGB images (np.ndarray)
            - "segmentation": list of corresponding masks (unchanged)
    image_names : list[str]
        List of image filenames (used only for iteration).
    reference_image : np.ndarray
        Reference image (RGB) used to standardize staining appearance.
    method : str, optional
        Stain normalization method. Must be either "macenko" or "vahadane".
        Default is "macenko".

    Returns
    -------
    dataset : dict
        The same dictionary as input, but with normalized RGB images.
        Masks remain unchanged.
    """

    # Validate method
    method = method.lower()
    if method not in {"macenko", "vahadane"}:
        raise ValueError(
            f"Invalid method '{method}'. Choose 'macenko' or 'vahadane'."
        )

    # Ensure reference image is in RGB
    if reference_image.shape[-1] == 3:
        # Convert if accidentally BGR (e.g. from cv2.imread)
        ref_img = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    else:
        ref_img = reference_image

    # Initialize and fit the normalizer
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(ref_img)

    # Apply normalization to each image in the dataset
    for i, name in enumerate(image_names):
        try:
            normalized_img = normalizer.transform(dataset["image"][i])
            dataset["image"][i] = normalized_img
        except Exception as e:
            print(f"[Warning] Skipping {name}: normalization failed ({e})")

    return dataset
