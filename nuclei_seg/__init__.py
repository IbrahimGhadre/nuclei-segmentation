# nuclei_seg/__init__.py
"""
nuclei_seg package
------------------

Modular deep learning framework for nuclei segmentation in H&E histopathology images.

This package provides all preprocessing, training, inference, and evaluation utilities
used by the main pipelines:

- training_pipeline.py
- evaluate_model.py
- test_pipeline.py

Modules include:
- Image and mask loading (.mat polygon annotations)
- Mirror padding and patch extraction
- Model training with MONAI + PyTorch (U-Net)
- Morphological post-processing and watershed segmentation
- Aggregated Jaccard Index (AJI) and nuclei counting metrics
"""
