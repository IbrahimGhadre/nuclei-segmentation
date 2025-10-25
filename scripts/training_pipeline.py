#!/usr/bin/env python3
# training_pipeline.py
#
# Train a lightweight U-Net (MONAI + PyTorch) for nuclei instance segmentation
# on H&E histology patches.
#
# Pipeline:
# 1. Load raw images + .mat polygon annotations (train / val)
# 2. Mirror-pad each full image and mask so they tile cleanly
# 3. Extract overlapping 256×256 patches (+ filter empty patches in train)
# 4. Build MONAI datasets with augmentations
# 5. Train UNet with BCE loss, monitor Dice on val
# 6. Save best checkpoint + qualitative snapshots + learning curves
#
# Repo layout this script expects:
#
# nuclei-segmentation/
# ├─ nuclei_seg/
# │   ├─ __init__.py
# │   ├─ lettura_set.py
# │   ├─ padding_fun.py
# │   ├─ divisione_in_patch.py
# │   ├─ unpatchify_and_unpadding.py      (not used during training, used later)
# │   ├─ label_instances.py
# │   ├─ watershed_fun.py
# │   ├─ aji_fun.py
# │   ├─ macenko.py
# │   ├─ normalize_staining.py
# ├─ training_pipeline.py   ← (this file)
# ├─ evaluate_model.py      ← (validation / metrics on full images)
# ├─ test_pipeline.py       ← (test set evaluation)
# ├─ README.md
# └─ requirements.txt
#
# Usage (example):
#   python training_pipeline.py \
#       --dataset-root ./DATASET \
#       --output-root ./experiments \
#       --patch-size 256 \
#       --epochs 20
#
# Expected dataset structure under --dataset-root:
#   DATASET/
#     train/
#       images/*.png (or .jpg)
#       manual/*.mat
#     val/
#       images/*.png
#       manual/*.mat
#
# The script will create (under --output-root):
#   experiments/
#     patches/train/images/*.jpg
#     patches/train/masks/*.jpg
#     patches/val/images/*.jpg
#     patches/val/masks/*.jpg
#     snapshots/<EXPERIMENT_NAME>/*visual samples*
#     best_metric_model.pth
#
# ------------------------------------------------------------------------------

import os
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from skimage.transform import rotate
import PIL.Image

import torch
import torch.nn as nn

from monai.transforms import (
    Compose,
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    ScaleIntensityRanged,
    AsDiscreted,
    RandFlipd,
    RandRotated,
    ToTensord,
    EnsureType,
    Activations,
)
from monai.data import (
    IterableDataset,
    PILReader,
    decollate_batch,
    DataLoader,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer
from monai.utils import set_determinism

# --- local project imports (our cleaned functions) ---
from nuclei_seg.lettura_set import lettura_set
from nuclei_seg.padding_fun import padding_fun
from nuclei_seg.divisione_in_patch import divisione_in_patch


# ------------------------------------------------------------------------------
# Config helper
# ------------------------------------------------------------------------------

class TrainConfig:
    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        patch_size: int = 256,
        overlap_frac: float = 0.5,
        batch_size: int = 32,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        bg_tolerance: float = 0.9,
        seed: int = 46,
        experiment_name: str = "UNET-EIM-256X256",
    ):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

        # raw dataset split dirs
        self.train_img_dir = self.dataset_root / "train" / "images"
        self.val_img_dir   = self.dataset_root / "val" / "images"

        # where we will write extracted patches
        self.patches_root = self.output_root / "patches"
        self.train_patch_img = self.patches_root / "train" / "images"
        self.train_patch_mask = self.patches_root / "train" / "masks"
        self.val_patch_img = self.patches_root / "val" / "images"
        self.val_patch_mask = self.patches_root / "val" / "masks"

        # snapshots of qualitative predictions
        self.snapshot_dir = self.output_root / "snapshots" / experiment_name

        # best model weights
        self.best_model_path = self.output_root / "best_metric_model.pth"

        # training hyperparams
        self.patch_size = patch_size
        self.overlap_frac = overlap_frac
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.bg_tolerance = bg_tolerance
        self.seed = seed


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_patch_lists(cfg: TrainConfig):
    """
    After we tile the padded images into patches on disk, build:
    - train_set_data: list[{"image": path, "segmentation": path}] with BG filtering
    - val_set_data:   same, no BG filtering
    """
    train_patches = sorted(os.listdir(cfg.train_patch_img))
    train_masks   = sorted(os.listdir(cfg.train_patch_mask))

    val_patches = sorted(os.listdir(cfg.val_patch_img))
    val_masks   = sorted(os.listdir(cfg.val_patch_mask))

    train_set_data = []
    for i in range(len(train_patches)):
        img_name = train_patches[i]

        # get corresponding mask filename (same suffix rule as original code)
        suffix = "_".join(img_name.split("_")[1:])          # e.g. "2_1_1.jpg"
        mask_name = "mask_" + suffix                        # e.g. "mask_2_1_1.jpg"

        mask_path = cfg.train_patch_mask / mask_name
        img_path  = cfg.train_patch_img  / img_name

        # load mask to measure background %
        m = cv2.imread(str(mask_path))
        if m is None:
            continue  # skip if something is weird
        m_bin = (m[:, :, 0] / 255).astype(np.uint8)

        zero_pixels = np.count_nonzero(m_bin == 0)
        total_pixels = cfg.patch_size * cfg.patch_size

        # keep patch only if foreground is at least (1-bg_tolerance)
        if zero_pixels < total_pixels * cfg.bg_tolerance:
            train_set_data.append({
                "image": str(img_path),
                "segmentation": str(mask_path),
            })

    # validation: keep all patches
    val_set_data = []
    for i in range(len(val_patches)):
        img_name = val_patches[i]
        suffix = "_".join(img_name.split("_")[1:])
        mask_name = "mask_" + suffix

        img_path  = cfg.val_patch_img  / img_name
        mask_path = cfg.val_patch_mask / mask_name

        val_set_data.append({
            "image": str(img_path),
            "segmentation": str(mask_path),
        })

    return train_set_data, val_set_data


def build_transforms():
    """
    Returns train_transforms, val_transforms (MONAI Compose objects)
    """
    train_transforms = Compose(
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

            # augmentations
            RandFlipd(keys=["image", "segmentation"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "segmentation"], spatial_axis=[1], prob=0.5),
            RandRotated(keys=["image", "segmentation"], range_x=0.5, prob=0.5),

            ToTensord(keys=["image", "segmentation"]),
        ]
    )

    val_transforms = Compose(
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

    return train_transforms, val_transforms


def validate_and_snapshot(
    model,
    val_loader,
    device,
    loss_fn,
    post_pred,
    post_label,
    snapshot_dir: Path,
    global_step: int,
    dice_metric_obj: DiceMetric,
):
    """
    Run inference on the validation loader:
    - compute Dice and BCE loss
    - save a qualitative snapshot (input | GT | prediction)
    """
    model.eval()
    step_losses = []
    dice_vals = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate", dynamic_ncols=True):
            val_inputs = batch["image"].to(device)
            val_labels = batch["segmentation"].to(device)

            # forward
            logits = model(val_inputs)  # raw
            loss_val = loss_fn(logits.sigmoid(), val_labels)
            step_losses.append(loss_val.item())

            # post-process for Dice
            val_outputs_list = [post_pred(o) for o in decollate_batch(logits)]
            val_labels_list  = [post_label(l) for l in decollate_batch(val_labels)]

            dice_metric_obj(y_pred=val_outputs_list, y=val_labels_list)
            dice_now = dice_metric_obj.aggregate().item()
            dice_vals.append(dice_now)
            dice_metric_obj.reset()

            # take first item only for visualization
            # input: (B,C,H,W) -> (H,W,C)
            vis_input = (
                val_inputs[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0
            )
            vis_gt = (
                val_labels_list[0][0].detach().cpu().numpy() * 255.0
            )
            vis_pred = (
                val_outputs_list[0][0].detach().cpu().numpy() * 255.0
            )

            # match the original notebook visualization
            vis_input = np.flipud(vis_input.astype(np.uint8))
            vis_gt = rotate(vis_gt, 90).astype(np.uint8)
            vis_pred = rotate(vis_pred, 90).astype(np.uint8)

            gt_rgb = np.stack([vis_gt]*3, axis=2)
            pr_rgb = np.stack([vis_pred]*3, axis=2)
            stack = np.concatenate([vis_input, gt_rgb, pr_rgb], axis=1).astype(np.uint8)

            # derive filename from patch path (strip extension)
            patch_name = os.path.basename(
                batch["image_meta_dict"]["filename_or_obj"][0]
            ).split(".")[0]

            this_step_dir = snapshot_dir / f"step_{global_step}"
            ensure_dir(this_step_dir)

            PIL.Image.fromarray(stack).save(
                this_step_dir / f"{patch_name}.png"
            )

    mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    return mean_dice, mean_loss


def train_loop(cfg: TrainConfig):
    """
    Full training loop:
    - load data
    - pad + patchify
    - build loaders
    - train UNet with BCE
    - run val each epoch
    - save best checkpoint + curves
    """

    # ------------------------------------------------------------------
    # 1. Prepare output dirs
    # ------------------------------------------------------------------
    ensure_dir(cfg.output_root)
    ensure_dir(cfg.patches_root)
    ensure_dir(cfg.train_patch_img)
    ensure_dir(cfg.train_patch_mask)
    ensure_dir(cfg.val_patch_img)
    ensure_dir(cfg.val_patch_mask)
    ensure_dir(cfg.snapshot_dir)

    # ------------------------------------------------------------------
    # 2. Load raw images + GT polygons and build bin masks
    # ------------------------------------------------------------------
    train_img_list = sorted(os.listdir(cfg.train_img_dir))
    val_img_list   = sorted(os.listdir(cfg.val_img_dir))

    training_data, _ = lettura_set(
        data_img=train_img_list,
        data_img_path=cfg.train_img_dir,
        data="train",
        base_dataset_dir=cfg.dataset_root,
    )

    validation_data, _ = lettura_set(
        data_img=val_img_list,
        data_img_path=cfg.val_img_dir,
        data="val",
        base_dataset_dir=cfg.dataset_root,
    )

    # ------------------------------------------------------------------
    # 3. Mirror padding
    # ------------------------------------------------------------------
    training_data, train_padding_info = padding_fun(
        cfg.patch_size,
        training_data,
        train_img_list,
    )
    validation_data, val_padding_info = padding_fun(
        cfg.patch_size,
        validation_data,
        val_img_list,
    )

    # ------------------------------------------------------------------
    # 4. Patch extraction
    # ------------------------------------------------------------------
    _patch_info_train = divisione_in_patch(
        data_img=train_img_list,
        dict_data=training_data,
        path_img=cfg.train_patch_img,
        path_mask=cfg.train_patch_mask,
        shape=cfg.patch_size,
        overlap=cfg.overlap_frac,
    )

    _patch_info_val = divisione_in_patch(
        data_img=val_img_list,
        dict_data=validation_data,
        path_img=cfg.val_patch_img,
        path_mask=cfg.val_patch_mask,
        shape=cfg.patch_size,
        overlap=cfg.overlap_frac,
    )

    # ------------------------------------------------------------------
    # 5. Build train/val patch lists
    # ------------------------------------------------------------------
    train_set_data, val_set_data = build_patch_lists(cfg)

    # ------------------------------------------------------------------
    # 6. Datasets + loaders
    # ------------------------------------------------------------------
    train_transforms, val_transforms = build_transforms()

    train_ds = IterableDataset(train_set_data, transform=train_transforms)
    val_ds   = IterableDataset(val_set_data,   transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 7. Model / loss / optimizer / metric / inferer
    # ------------------------------------------------------------------
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

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    post_pred  = Compose([EnsureType(), Activations(sigmoid=True), AsDiscreted(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscreted(threshold=0.5)])

    dice_metric_obj = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False,
    )

    inferer = SimpleInferer()

    set_determinism(seed=cfg.seed)
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # 8. Training loop with per-epoch validation
    # ------------------------------------------------------------------
    epoch_loss_values = []
    epoch_val_loss_values = []
    metric_values = []

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        steps = 0

        prog = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]", dynamic_ncols=True)
        for batch in prog:
            x = batch["image"].to(device)
            y = batch["segmentation"].to(device)

            optimizer.zero_grad()
            logits = model(x).sigmoid()
            loss = loss_function(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            prog.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, steps)

        # ---- validation and snapshot (once per epoch)
        dice_val, loss_val = validate_and_snapshot(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_function,
            post_pred=post_pred,
            post_label=post_label,
            snapshot_dir=cfg.snapshot_dir,
            global_step=epoch + 1,
            dice_metric_obj=dice_metric_obj,
        )

        epoch_loss_values.append(avg_train_loss)
        epoch_val_loss_values.append(loss_val)
        metric_values.append(dice_val)

        if dice_val > best_dice:
            best_dice = dice_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), cfg.best_model_path)
            print(f"[SAVE] epoch {epoch+1}: Dice improved to {best_dice:.4f} -> {cfg.best_model_path}")
        else:
            print(f"[KEEP] epoch {epoch+1}: Dice={dice_val:.4f} (best {best_dice:.4f} @ epoch {best_epoch})")

    print(f"Training done. Best Dice={best_dice:.4f} at epoch {best_epoch}.")
    print(f"Best weights saved to: {cfg.best_model_path}")
    print(f"Validation snapshots saved under: {cfg.snapshot_dir}")

    # ------------------------------------------------------------------
    # 9. Plot curves and save them
    # ------------------------------------------------------------------
    fig = plt.figure("training_curves", figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("BCE Loss (train vs val)")
    x_axis = list(range(1, len(epoch_loss_values) + 1))
    plt.plot(x_axis, epoch_loss_values, label="train BCE")
    plt.plot(x_axis, epoch_val_loss_values, label="val BCE")
    plt.xlabel("epoch")
    plt.legend(loc="upper right")

    plt.subplot(1, 2, 2)
    plt.title("Validation Dice")
    plt.plot(x_axis, metric_values, label="Dice (val)")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")

    fig.tight_layout()

    curve_path = cfg.output_root / "training_curves.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)

    print(f"Saved training curves to: {curve_path}")


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train nuclei segmentation U-Net.")
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to DATASET/ (with train/images, train/manual, val/images, val/manual).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="./experiments",
        help="Where to write patches/, snapshots/, best model, curves.",
    )
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--overlap-frac", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bg-tolerance", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=46)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        patch_size=args.patch_size,
        overlap_frac=args.overlap_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        bg_tolerance=args.bg_tolerance,
        seed=args.seed,
    )

    train_loop(cfg)
