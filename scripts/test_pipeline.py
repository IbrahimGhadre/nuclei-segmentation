"""
test_pipeline.py
Evaluate a trained nuclei segmentation model on the TEST split.

Steps:
1. Load test images + GT nuclei polygons (.mat) → build binary GT masks
2. Mirror-pad and tile into overlapping patches
3. Run inference with trained U-Net (MONAI)
4. Stitch patches back to full-res masks (unpatchify + remove padding)
5. Post-process (opening, optional watershed)
6. Compute metrics (Dice on patches, AJI and nuclei count error on full images)
7. Export predicted nuclei as polygons (.mat) using mask2poly()

"""

import os
import numpy as np
import cv2
from tqdm.auto import tqdm
from skimage.transform import rotate
from scipy.ndimage import binary_opening  # alternative to cv2.morphologyEx if you want

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    ScaleIntensityRanged,
    AsDiscreted,
    ToTensord,
    Activations,
    EnsureType,
)
from monai.data import IterableDataset, PILReader, decollate_batch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer

# ---- local project utils (our cleaned modules)
from src.lettura_set import lettura_set
from src.padding_fun import padding_fun
from src.divisione_in_patch import divisione_in_patch
from src.unpatchify_and_unpadding import unpatchify_and_unpadding
from src.label_instances import label_instances  # nuclei labeling
from src.watershed_fun import apply_watershed
from src.aji_fun import aji_fun
from src.mask2poly import mask2poly


# =====================
# 0. Config / paths
# =====================

PROJECT_ROOT = "."  # adjust if needed
DATASET_ROOT = os.path.join(PROJECT_ROOT, "DATASET")

TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")
TEST_MANUAL_DIR = os.path.join(DATASET_ROOT, "test", "manual")

# patch dirs
TEST_PATCH_ROOT = os.path.join(PROJECT_ROOT, "test_pre_rete")
TEST_PATCH_IMG_DIR = os.path.join(TEST_PATCH_ROOT, "images")
TEST_PATCH_MASK_DIR = os.path.join(TEST_PATCH_ROOT, "masks")

# inference dumps for visualization
STACK_DIR = os.path.join(PROJECT_ROOT, "testing_post_rete")
STACK_PRED_DIR = os.path.join(PROJECT_ROOT, "testing_post_rete_pred")

# final coordinate export
COORD_DIR = os.path.join(PROJECT_ROOT, "coordinate", "test")

# trained weights
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_metric_model.pth")

# hyperparams
DIM_PATCH = 256
OVERLAP_FRAC = 0.5
BATCH_SIZE = 32
NUM_WORKERS = 0  # can increase on local workstation
BACKGROUND_TOLERANCE = 0.9  # not actually used in test, we keep all patches


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


ensure_dir(TEST_PATCH_IMG_DIR)
ensure_dir(TEST_PATCH_MASK_DIR)
ensure_dir(STACK_DIR)
ensure_dir(STACK_PRED_DIR)
ensure_dir(COORD_DIR)


# =====================
# 1. Load TEST set and build GT masks
# =====================

# list of image filenames in test/images
test_img_list = sorted(os.listdir(TEST_IMG_DIR))

testing_data, N_gt_test = lettura_set(
    data_img=test_img_list,
    data_img_path=TEST_IMG_DIR,
    data="test",                 # split name for .mat lookup
    base_dataset_dir=DATASET_ROOT,
)

# we’ll also need test_img_manual names (the .mat files) for AJI and nuclei count
test_img_manual = [f"{name[:-4]}.mat" for name in test_img_list]


# =====================
# 2. Padding
# =====================

testing_data, test_padding = padding_fun(
    shape=DIM_PATCH,
    dict_data=testing_data,
    data_img=test_img_list,
)


# =====================
# 3. Patch extraction (256x256 with 50% overlap)
# =====================

patch_info_test = divisione_in_patch(
    data_img=test_img_list,
    dict_data=testing_data,
    path_img=TEST_PATCH_IMG_DIR,
    path_mask=TEST_PATCH_MASK_DIR,
    shape=DIM_PATCH,
    overlap=OVERLAP_FRAC,
)

# prepare MONAI-style item list for the test loader:
test_patches = sorted(os.listdir(TEST_PATCH_IMG_DIR))

test_set_data = []
for patch_name in test_patches:
    # The mask patch has same suffix after "immagine_"
    suffix = "_".join(patch_name.split("_")[1:])
    mask_name = f"mask_{suffix}"

    sample = {
        "image": os.path.join(TEST_PATCH_IMG_DIR, patch_name),
        "segmentation": os.path.join(TEST_PATCH_MASK_DIR, mask_name),
    }
    test_set_data.append(sample)


# =====================
# 4. Dataloader (+ transforms)
# =====================

test_transforms = Compose(
    [
        LoadImaged(
            keys=["image", "segmentation"],
            image_only=False,
            reader=PILReader(),
        ),
        AsChannelFirstd(keys=["image"]),
        AddChanneld(keys=["segmentation"]),
        ScaleIntensityRanged(
            keys=["image", "segmentation"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        AsDiscreted(keys=["segmentation"], threshold=0.5),
        ToTensord(keys=["image", "segmentation"]),
    ]
)

test_ds = IterableDataset(data=test_set_data, transform=test_transforms)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# =====================
# 5. Load model
# =====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

loss_function = nn.BCELoss()
inferer = SimpleInferer()

post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscreted(threshold=0.5)])
post_label = Compose([EnsureType(), AsDiscreted(threshold=0.5)])

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


# =====================
# 6. Inference on patches + Dice on patches
#    Also save visual triplets and raw pred masks.
# =====================

dice_list = []
epoch_test_loss_vals = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing (patch-level)", dynamic_ncols=True):
        imgs = batch["image"].to(device)          # [B, 3, H, W]
        gts = batch["segmentation"].to(device)    # [B, 1, H, W]

        logits = inferer(imgs, model)             # raw (B,1,H,W)
        loss_val = loss_function(logits.sigmoid(), gts)
        epoch_test_loss_vals.append(loss_val.item())

        # binarize preds and gts for dice
        preds_list = [post_pred(i) for i in decollate_batch(logits)]
        gts_list   = [post_label(i) for i in decollate_batch(gts)]

        dice_metric(y_pred=preds_list, y=gts_list)
        dice_now = dice_metric.aggregate().item()
        dice_list.append(dice_now)
        dice_metric.reset()

        # For each item in batch: save visualization and predicted mask
        for b_idx in range(len(preds_list)):
            # convert tensors -> numpy uint8
            img_np = imgs[b_idx].detach().cpu().permute(1, 2, 0).numpy() * 255.0
            gt_np  = gts_list[b_idx][0].detach().cpu().numpy() * 255.0
            pr_np  = preds_list[b_idx][0].detach().cpu().numpy() * 255.0

            # match the original notebook’s orientation tricks
            img_np = np.flipud(img_np.astype(np.uint8))
            gt_np  = rotate(gt_np, 90).astype(np.uint8)
            pr_np  = rotate(pr_np, 90).astype(np.uint8)

            gt_rgb = np.stack([gt_np]*3, axis=2)
            pr_rgb = np.stack([pr_np]*3, axis=2)

            stack = np.concatenate([img_np, gt_rgb, pr_rgb], axis=1).astype(np.uint8)

            # derive filename from the patch path
            patch_path = batch["image_meta_dict"]["filename_or_obj"][b_idx]
            base_name = os.path.basename(patch_path).split(".")[0]

            # save triplet stack
            out_stack_path = os.path.join(STACK_DIR, f"{base_name}.png")

            # save predicted mask alone (after flip/rotate we used for display)
            out_pred_path = os.path.join(STACK_PRED_DIR, f"{base_name}.png")

            import PIL.Image
            PIL.Image.fromarray(stack).save(out_stack_path)
            # we save pr_np flipped the same way we visualized
            PIL.Image.fromarray(np.flipud(pr_np)).save(out_pred_path)

mean_dice_test = float(np.mean(dice_list))
std_dice_test = float(np.std(dice_list))
mean_loss_test = float(np.mean(epoch_test_loss_vals))

print(f"[PATCH Dice] {mean_dice_test:.4f} ± {std_dice_test:.4f}")
print(f"[PATCH BCE ] {mean_loss_test:.4f}")


# =====================
# 7. Recompose full images back to original resolution
# =====================

# unpatchify_and_unpadding expects:
#   - lista dei nomi immagini originali
#   - cartella con le predizioni patch-level (STACK_PRED_DIR)
#   - padding info
#   - info sul grid di patch
#   - dim_patch/2 as patch_size_crop (128 if patch=256)

full_pred_masks = unpatchify_and_unpadding(
    data_img=test_img_list,
    path_mask_post_rete=STACK_PRED_DIR,
    padding=test_padding,
    dizionario_patches=patch_info_test,
    patch_size_crop=int(DIM_PATCH / 2),
)

# post-processing: morphological opening (like original script)
kernel = np.ones((3, 3), np.uint8)
full_pred_open = [
    cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    for m in full_pred_masks
]


# =====================
# 8. Compute AJI and nuclei count errors on full images
# =====================

# 8a. AJI post rete
labeled_rete_test, features_rete_test = label_instances(full_pred_masks)

aji_rete = []
for idx, lbl_map in enumerate(labeled_rete_test):
    gt_mat_path = os.path.join(TEST_MANUAL_DIR, test_img_manual[idx])
    aji_val = aji_fun(gt_mat_path, lbl_map)
    aji_rete.append(aji_val)

aji_rete_mean = float(np.mean(aji_rete))
aji_rete_std = float(np.std(aji_rete))
print(f"[IMG AJI post rete] {aji_rete_mean:.4f} ± {aji_rete_std:.4f}")

# 8b. AJI post opening
labeled_open_test, features_open_test = label_instances(full_pred_open)

aji_open = []
for idx, lbl_map in enumerate(labeled_open_test):
    gt_mat_path = os.path.join(TEST_MANUAL_DIR, test_img_manual[idx])
    aji_val = aji_fun(gt_mat_path, lbl_map)
    aji_open.append(aji_val)

aji_open_mean = float(np.mean(aji_open))
aji_open_std = float(np.std(aji_open))
print(f"[IMG AJI post opening] {aji_open_mean:.4f} ± {aji_open_std:.4f}")

# nuclei count error
err_count_rete = []
err_count_open = []
for i in range(len(test_img_list)):
    err_count_rete.append(abs(N_gt_test[i] - features_rete_test[i]))
    err_count_open.append(abs(N_gt_test[i] - features_open_test[i]))

print(
    f"[Count Error rete] {np.mean(err_count_rete):.2f} ± {np.std(err_count_rete):.2f}"
)
print(
    f"[Count Error open] {np.mean(err_count_open):.2f} ± {np.std(err_count_open):.2f}"
)


# =====================
# 9. Export nuclei polygons (.mat) for final submission
# =====================

# We export from the opened masks (post-processing).
# mask2poly will create one .mat file per test image, each containing contours.

mask2poly(
    set_data=full_pred_open,
    data_img_names=test_img_list,
    output_dir=COORD_DIR,
)

print(f"Done. Predictions (patch-level) in {STACK_PRED_DIR},")
print(f"full-res masks reconstructed, metrics computed, and polygons saved to {COORD_DIR}.")
