#!/usr/bin/env python3
# evaluate_model.py
#
# Evaluate a trained nuclei segmentation model on TRAIN and VAL sets.
#
# Pipeline:
# 1. Recreate patches (same preprocessing as training) for train and val.
# 2. Load the best trained model (best_metric_model.pth) and run inference
#    patch-by-patch.
#    - Save each predicted patch as an individual image (one per patch).
# 3. Reconstruct full-size masks from predicted patches using
#    unpatchify_and_unpadding.
# 4. Post-processing on reconstructed masks:
#    - morphological opening
#    - marker-controlled watershed
# 5. Compute metrics:
#    - Mean Dice on patches (train/val)
#    - Mean AJI for: raw network output, after opening, after opening+watershed
#    - Nuclei count error with respect to GT (|#GT - #pred|) for each version
#
# Final output: print mean/std metrics for train and val.

import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm.auto import tqdm
from skimage.transform import rotate
from scipy.ndimage import binary_opening

import torch
import torch.nn as nn

from monai.data import IterableDataset, DataLoader, PILReader, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    ScaleIntensityRanged,
    AsDiscreted,
    ToTensord,
    EnsureType,
    Activations,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer
from monai.utils import set_determinism

# --- local utilities from your cleaned package nuclei_seg/ ---
from nuclei_seg.lettura_set import lettura_set
from nuclei_seg.padding_fun import padding_fun
from nuclei_seg.divisione_in_patch import divisione_in_patch
from nuclei_seg.unpatchify_and_unpadding import unpatchify_and_unpadding
from nuclei_seg.watershed_fun import apply_watershed
from nuclei_seg.label_instances import label_instances
from nuclei_seg.aji_fun import aji_fun


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

class EvalConfig:
    def __init__(
        self,
        dataset_root: Path,
        output_root: Path,
        patch_size: int = 256,
        overlap_frac: float = 0.5,
        batch_size: int = 32,
        seed: int = 46,
        model_path: Path | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

        # original dataset dirs
        self.train_img_dir = self.dataset_root / "train" / "images"
        self.val_img_dir   = self.dataset_root / "val"   / "images"
        self.train_manual_dir = self.dataset_root / "train" / "manual"
        self.val_manual_dir   = self.dataset_root / "val"   / "manual"

        # dirs where training_pipeline put the patches
        # (we will reuse the same layout to be consistent)
        self.patches_root = self.output_root / "patches"
        self.train_patch_img = self.patches_root / "train" / "images"
        self.train_patch_mask = self.patches_root / "train" / "masks"
        self.val_patch_img = self.patches_root / "val" / "images"
        self.val_patch_mask = self.patches_root / "val" / "masks"

        # dirs to store network predictions on each patch
        self.train_pred_patch_dir = self.output_root / "training_post_rete_pred"
        self.val_pred_patch_dir   = self.output_root / "validation_post_rete_pred"

        # device / loader params
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlap_frac = overlap_frac
        self.seed = seed

        # model weights path
        if model_path is None:
            self.model_path = self.output_root / "best_metric_model.pth"
        else:
            self.model_path = Path(model_path)

        # used later for counting nuclei error
        self.background_label_val = 0  # background assumed 0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 1. Dataset tiling and loader creation (same as training)
# ------------------------------------------------------------------------------

def build_patch_lists_for_eval(cfg: EvalConfig):
    """
    Build lists of dicts [{"image":..., "segmentation":...}] for train and val.
    Here we KEEP all patches (no background filtering), because for evaluation
    we want full spatial coverage.
    """
    train_patches = sorted(os.listdir(cfg.train_patch_img))
    val_patches   = sorted(os.listdir(cfg.val_patch_img))

    train_list = []
    for img_name in train_patches:
        suffix = "_".join(img_name.split("_")[1:])
        mask_name = "mask_" + suffix
        train_list.append({
            "image": str(cfg.train_patch_img / img_name),
            "segmentation": str(cfg.train_patch_mask / mask_name),
        })

    val_list = []
    for img_name in val_patches:
        suffix = "_".join(img_name.split("_")[1:])
        mask_name = "mask_" + suffix
        val_list.append({
            "image": str(cfg.val_patch_img / img_name),
            "segmentation": str(cfg.val_patch_mask / mask_name),
        })

    return train_list, val_list


def build_eval_transforms():
    """
    Evaluation transforms:
    - load image/mask
    - channel-first for image, add channel for mask
    - scale [0,255] -> [0,1]
    - binarize mask
    - to tensor
    (no augmentation)
    """
    eval_transforms = Compose(
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
                a_min=0, a_max=255,
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            AsDiscreted(keys=["segmentation"], threshold=0.5),
            ToTensord(keys=["image", "segmentation"]),
        ]
    )
    return eval_transforms


def prepare_data_and_patches(cfg: EvalConfig):
    """
    Reproduce preprocessing exactly like in training:
    - lettura_set -> (img RGB list, boolean mask list)
    - padding_fun -> mirror padding + record padding info
    - divisione_in_patch -> tile into overlapping patches and save them to disk
    - build_patch_lists_for_eval -> build per-patch dicts for MONAI
    Also returns:
      * per-image padding info
      * per-image patch grid info
    which are required later to reconstruct full-size masks.
    """
    # ensure output dirs exist
    ensure_dir(cfg.train_patch_img)
    ensure_dir(cfg.train_patch_mask)
    ensure_dir(cfg.val_patch_img)
    ensure_dir(cfg.val_patch_mask)
    ensure_dir(cfg.train_pred_patch_dir)
    ensure_dir(cfg.val_pred_patch_dir)

    # list original image names
    train_img_list = sorted(os.listdir(cfg.train_img_dir))
    val_img_list   = sorted(os.listdir(cfg.val_img_dir))

    # load GT masks (bool) using lettura_set
    train_dict, train_gt_counts = lettura_set(
        data_img=train_img_list,
        data_img_path=cfg.train_img_dir,
        data="train",
        base_dataset_dir=cfg.dataset_root,
    )
    val_dict, val_gt_counts = lettura_set(
        data_img=val_img_list,
        data_img_path=cfg.val_img_dir,
        data="val",
        base_dataset_dir=cfg.dataset_root,
    )

    # mirror padding
    train_dict, train_padding = padding_fun(cfg.patch_size, train_dict, train_img_list)
    val_dict, val_padding     = padding_fun(cfg.patch_size, val_dict,   val_img_list)

    # patchify to disk
    train_patch_info = divisione_in_patch(
        data_img=train_img_list,
        dict_data=train_dict,
        path_img=cfg.train_patch_img,
        path_mask=cfg.train_patch_mask,
        shape=cfg.patch_size,
        overlap=cfg.overlap_frac,
    )
    val_patch_info = divisione_in_patch(
        data_img=val_img_list,
        dict_data=val_dict,
        path_img=cfg.val_patch_img,
        path_mask=cfg.val_patch_mask,
        shape=cfg.patch_size,
        overlap=cfg.overlap_frac,
    )

    # build per-patch dataset lists for MONAI
    train_list, val_list = build_patch_lists_for_eval(cfg)

    return (
        train_img_list,
        val_img_list,
        train_gt_counts,
        val_gt_counts,
        train_padding,
        val_padding,
        train_patch_info,
        val_patch_info,
        train_list,
        val_list,
    )


# ------------------------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------------------------

def load_trained_model(cfg: EvalConfig, device: torch.device):
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    state = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------------------------
# 3. Patch-level inference + Dice computation + saving predicted patches
# ------------------------------------------------------------------------------

def run_inference_and_save_patches(
    cfg: EvalConfig,
    loader: DataLoader,
    device: torch.device,
    model: UNet,
    post_pred,
    post_label,
    dice_metric_obj,
    pred_out_dir: Path,
    split_name: str,
):
    """
    - Compute mean Dice on a DataLoader (patch-level)
    - Compute mean BCE loss on patches
    - For each patch, save the binarized prediction (after sigmoid+0.5)
      as a PNG in pred_out_dir.
      Output filename keeps the "immagine_i_j_k" convention.
    Returns:
        mean_dice, mean_loss
    """
    ensure_dir(pred_out_dir)

    loss_fn = nn.BCELoss()

    dice_vals = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference {split_name}", dynamic_ncols=True):
            imgs = batch["image"].to(device)
            gts  = batch["segmentation"].to(device)

            logits = model(imgs)  # (B,1,H,W)
            loss_val = loss_fn(logits.sigmoid(), gts)
            losses.append(loss_val.item())

            # post-process to binary tensors
            preds_list = [post_pred(o) for o in decollate_batch(logits)]
            gts_list   = [post_label(l) for l in decollate_batch(gts)]

            dice_metric_obj(y_pred=preds_list, y=gts_list)
            dice_now = dice_metric_obj.aggregate().item()
            dice_vals.append(dice_now)
            dice_metric_obj.reset()

            # save each predicted patch mask
            for b in range(len(preds_list)):
                pred_np = preds_list[b][0].detach().cpu().numpy().astype(np.uint8) * 255

                # original patch filename from metadata
                patch_path = batch["image_meta_dict"]["filename_or_obj"][b]
                # e.g. ".../immagine_3_5_2.jpg"
                base_jpg = os.path.basename(patch_path)

                out_name = os.path.splitext(base_jpg)[0] + ".png"

                cv2.imwrite(
                    str(pred_out_dir / out_name),
                    pred_np,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )

    mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
    mean_loss = float(np.mean(losses))    if losses    else 0.0
    return mean_dice, mean_loss


# ------------------------------------------------------------------------------
# 4. Full-image reconstruction + post-processing + AJI / count error
# ------------------------------------------------------------------------------

def morph_opening_list(masks, kernel_size=3, iterations=1):
    """
    Apply morphological opening (erode+dilate) to each mask in `masks`.
    Returns a list of uint8 masks.
    """
    k = np.ones((kernel_size, kernel_size), np.uint8)
    out = []
    for m in masks:
        opened = cv2.morphologyEx(
            m.astype(np.uint8),
            cv2.MORPH_OPEN,
            k,
            iterations=iterations
        )
        out.append(opened)
    return out


def compute_aji_and_count_errors(
    split_name: str,
    img_names: list[str],
    manual_dir: Path,
    reconstructed_masks_postrete: list[np.ndarray],
    reconstructed_masks_open: list[np.ndarray],
    reconstructed_masks_openws: list[np.ndarray],
    gt_counts: list[int],
):
    """
    Compute:
    - AJI after raw network prediction ("post rete")
    - AJI after morphological opening
    - AJI after opening + watershed
    Also compute nuclei counting error |GT - predicted_instances| for each version.
    """
    # label connected components for each post-processing variant
    labeled_rete, features_rete = label_instances(reconstructed_masks_postrete)
    labeled_open, features_open = label_instances(reconstructed_masks_open)
    labeled_openws, features_openws = label_instances(reconstructed_masks_openws)

    aji_rete_list = []
    aji_open_list = []
    aji_openws_list = []

    err_rete = []
    err_open = []
    err_openws = []

    for idx, img_name in enumerate(img_names):
        base_noext = os.path.splitext(img_name)[0]
        mat_path = manual_dir / f"{base_noext}.mat"

        aji_r = aji_fun(str(mat_path), labeled_rete[idx])
        aji_o = aji_fun(str(mat_path), labeled_open[idx])
        aji_w = aji_fun(str(mat_path), labeled_openws[idx])

        aji_rete_list.append(aji_r)
        aji_open_list.append(aji_o)
        aji_openws_list.append(aji_w)

        err_rete.append(abs(gt_counts[idx] - features_rete[idx]))
        err_open.append(abs(gt_counts[idx] - features_open[idx]))
        err_openws.append(abs(gt_counts[idx] - features_openws[idx]))

    results = {
        "aji_rete_mean":      float(np.mean(aji_rete_list)),
        "aji_rete_std":       float(np.std(aji_rete_list)),
        "aji_open_mean":      float(np.mean(aji_open_list)),
        "aji_open_std":       float(np.std(aji_open_list)),
        "aji_openws_mean":    float(np.mean(aji_openws_list)),
        "aji_openws_std":     float(np.std(aji_openws_list)),
        "err_rete_mean":      float(np.mean(err_rete)),
        "err_rete_std":       float(np.std(err_rete)),
        "err_open_mean":      float(np.mean(err_open)),
        "err_open_std":       float(np.std(err_open)),
        "err_openws_mean":    float(np.mean(err_openws)),
        "err_openws_std":     float(np.std(err_openws)),
    }

    print(f"\n=== [{split_name.upper()} SET] ===")
    print(f"AJI post rete:           {results['aji_rete_mean']:.4f} ± {results['aji_rete_std']:.4f}")
    print(f"AJI post opening:        {results['aji_open_mean']:.4f} ± {results['aji_open_std']:.4f}")
    print(f"AJI post opening+WS:     {results['aji_openws_mean']:.4f} ± {results['aji_openws_std']:.4f}")
    print(f"Nuclei error post rete:  {results['err_rete_mean']:.2f} ± {results['err_rete_std']:.2f}")
    print(f"Nuclei error opening:    {results['err_open_mean']:.2f} ± {results['err_open_std']:.2f}")
    print(f"Nuclei error open+WS:    {results['err_openws_mean']:.2f} ± {results['err_openws_std']:.2f}")

    return results


def reconstruct_full_images_and_metrics(
    cfg: EvalConfig,
    img_list: list[str],
    padding_info: dict,
    patch_info: dict,
    pred_dir: Path,
    patch_size_crop: int,
    split_name: str,
    manual_dir: Path,
    gt_counts: list[int],
):
    """
    1. Reconstruct full-resolution masks using unpatchify_and_unpadding.
    2. Apply morphological opening.
    3. Apply watershed.
    4. Compute AJI and nuclei counting error.
    """
    # 1. reconstruction
    recon_masks = unpatchify_and_unpadding(
        data_img=img_list,
        path_mask_post_rete=str(pred_dir),
        padding=padding_info,
        dizionario_patches=patch_info,
        patch_size_crop=patch_size_crop,
    )

    # cast to uint8 [0..255]
    recon_masks_u8 = [m.astype(np.uint8) for m in recon_masks]

    # 2. morphological opening
    opened_masks = morph_opening_list(recon_masks_u8, kernel_size=3, iterations=1)

    # 3. watershed
    ws_masks = apply_watershed(opened_masks)

    # 4. AJI + nuclei error
    results = compute_aji_and_count_errors(
        split_name=split_name,
        img_names=img_list,
        manual_dir=manual_dir,
        reconstructed_masks_postrete=recon_masks_u8,
        reconstructed_masks_open=opened_masks,
        reconstructed_masks_openws=ws_masks,
        gt_counts=gt_counts,
    )

    return results


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate nuclei segmentation model on train/val sets."
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to DATASET/ with train/images, train/manual, val/images, val/manual",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="./experiments",
        help="Folder where patches/, best_metric_model.pth, etc. live.",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size used in training (e.g. 256).",
    )
    p.add_argument(
        "--overlap-frac",
        type=float,
        default=0.5,
        help="Patch overlap fraction (0.5 -> stride 128 if patch=256).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=46,
        help="Random seed.",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to model weights (default: <output-root>/best_metric_model.pth)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg = EvalConfig(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        patch_size=args.patch_size,
        overlap_frac=args.overlap_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        model_path=Path(args.model_path) if args.model_path else None,
    )

    set_determinism(seed=cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. Prepare data and patches exactly like training
    (
        train_img_list,
        val_img_list,
        train_gt_counts,
        val_gt_counts,
        train_padding,
        val_padding,
        train_patch_info,
        val_patch_info,
        train_list,
        val_list,
    ) = prepare_data_and_patches(cfg)

    # 1. Create eval DataLoaders (no augmentation)
    eval_transforms = build_eval_transforms()

    train_ds = IterableDataset(train_list, transform=eval_transforms)
    val_ds   = IterableDataset(val_list,   transform=eval_transforms)

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

    # 2. Load trained model weights
    model = load_trained_model(cfg, device)

    # 3. Define post-processing for predictions/labels + Dice metric
    post_pred  = Compose([EnsureType(), Activations(sigmoid=True), AsDiscreted(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscreted(threshold=0.5)])
    dice_metric_obj = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # 4. Patch-level inference + save predicted patches
    print("\n[TRAIN SPLIT - patch-level inference]")
    train_dice_patch, train_loss_patch = run_inference_and_save_patches(
        cfg=cfg,
        loader=train_loader,
        device=device,
        model=model,
        post_pred=post_pred,
        post_label=post_label,
        dice_metric_obj=dice_metric_obj,
        pred_out_dir=cfg.train_pred_patch_dir,
        split_name="train",
    )
    print(f"Train patch Dice mean: {train_dice_patch:.4f}")
    print(f"Train patch BCE mean:  {train_loss_patch:.4f}")

    print("\n[VAL SPLIT - patch-level inference]")
    val_dice_patch, val_loss_patch = run_inference_and_save_patches(
        cfg=cfg,
        loader=val_loader,
        device=device,
        model=model,
        post_pred=post_pred,
        post_label=post_label,
        dice_metric_obj=dice_metric_obj,
        pred_out_dir=cfg.val_pred_patch_dir,
        split_name="val",
    )
    print(f"Val patch Dice mean: {val_dice_patch:.4f}")
    print(f"Val patch BCE mean:  {val_loss_patch:.4f}")

    # 5. Full reconstruction + AJI / nuclei count error
    print("\n[Reconstruct TRAIN images + compute AJI / nuclei count error]")
    train_results = reconstruct_full_images_and_metrics(
        cfg=cfg,
        img_list=train_img_list,
        padding_info=train_padding,
        patch_info=train_patch_info,
        pred_dir=cfg.train_pred_patch_dir,
        patch_size_crop=cfg.patch_size // 2,  # e.g. 128 if patch=256
        split_name="train",
        manual_dir=cfg.train_manual_dir,
        gt_counts=train_gt_counts,
    )

    print("\n[Reconstruct VAL images + compute AJI / nuclei count error]")
    val_results = reconstruct_full_images_and_metrics(
        cfg=cfg,
        img_list=val_img_list,
        padding_info=val_padding,
        patch_info=val_patch_info,
        pred_dir=cfg.val_pred_patch_dir,
        patch_size_crop=cfg.patch_size // 2,
        split_name="val",
        manual_dir=cfg.val_manual_dir,
        gt_counts=val_gt_counts,
    )

    print("\n=== FINAL SUMMARY ===")
    print(f"Patch Dice (train): {train_dice_patch:.4f}")
    print(f"Patch Dice (val):   {val_dice_patch:.4f}")
    print("\nTrain AJI post net / opening / opening+WS:",
          train_results['aji_rete_mean'],
          train_results['aji_open_mean'],
          train_results['aji_openws_mean'])
    print("Val AJI post net / opening / opening+WS:",
          val_results['aji_rete_mean'],
          val_results['aji_open_mean'],
          val_results['aji_openws_mean'])
    print("\nTrain nuclei error (open+WS):",
          train_results['err_openws_mean'], "±", train_results['err_openws_std'])
    print("Val nuclei error (open+WS):",
          val_results['err_openws_mean'], "±", val_results['err_openws_std'])


if __name__ == "__main__":
    main()
