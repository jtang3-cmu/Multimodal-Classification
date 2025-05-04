"""Train script for multimodal AMD classification – **FAST MODE**

Trains the model **only on a random subset of batches each epoch** to speed up
iterations.

Key flags
-----------
* `--train_frac`  Fraction of training *batches* to use every epoch (0 < f ≤ 1).
* `--max_train_batches`  Absolute cap on batches per epoch (overrides
  `--train_frac` when set).
* Default behaviour (`--train_frac 1.0`) keeps the full‑dataset training loop.

This file handles **training + validation only**.  Use **eval.py** for
held‑out test‑set evaluation.
"""

import os
import argparse
import time
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset import MultimodalAMDDataset
from model import create_model
from utils import set_seed, plot_training_history

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal AMD Training – train only; use eval.py for inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save checkpoints & plots")

    # Model architecture
    parser.add_argument("--model_type", type=str, default="multimodal", choices=["multimodal", "image_only", "tabular_only"], help="Backbone variant")
    parser.add_argument("--image_encoder_type", type=str, default="resnet50", choices=["resnet50", "retfound"], help="CNN encoder")
    parser.add_argument("--retfound_weights", type=str, default="RETFound_MAE/RETFound_mae_natureOCT.pth", help="Path to RETFound weights")
    parser.add_argument("--freeze_encoders", action="store_true", help="Freeze encoders during training")
    parser.add_argument("--tab_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)

    # Optimisation
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of volumes for validation")
    parser.add_argument("--seed", type=int, default=42)

    # Fast‑mode controls
    parser.add_argument("--train_frac", type=float, default=1.0, help="Fraction of training *batches* to use each epoch (0 < f ≤ 1)")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Absolute max #batches per epoch (overrides --train_frac if set)")

    # Data sources (hard‑coded paths for now)
    parser.add_argument("--anno_ori", type=str, default=r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_ori.xlsx")
    parser.add_argument("--imgs_ori", type=str, default=r"D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data")
    parser.add_argument("--anno_new", type=str, default=r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_new.xlsx")
    parser.add_argument("--imgs_new", type=str, default=r"D:/cleaning_GUI_annotated_Data/New_Data")

    # TabTransformer fine‑tune only (optional shortcut)
    parser.add_argument("--tune_tab", action="store_true", help="Fine‑tune TabTransformer only and exit")
    parser.add_argument("--tab_data_path", type=str, default="annotation_modified_final_forTrain_v3.xlsx")
    parser.add_argument("--tab_epochs", type=int, default=20)
    parser.add_argument("--tab_batch_size", type=int, default=32)

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Helper – validation step
# -----------------------------------------------------------------------------

def _evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, args) -> Tuple[float, float]:
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            if args.model_type == "multimodal":
                outputs = model(batch["image"].to(device), batch["categorical"].to(device), batch["continuous"].to(device))
                labels = batch["label"].to(device)
            elif args.model_type == "image_only":
                outputs = model(batch["image"].to(device))
                labels = batch["label"].to(device)
            else:  # tabular_only
                outputs = model(batch["categorical"].to(device), batch["continuous"].to(device))
                labels = batch["label"].to(device)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return float(np.mean(losses)), float(accuracy_score(targets, preds))

# -----------------------------------------------------------------------------
# Core training loop (subset‑of‑batches + best‑ckpt save)
# -----------------------------------------------------------------------------

def train_model(
    args,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, List[float]], str]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"]}
    best_acc, best_path = -1.0, ""

    total_batches = len(train_loader)
    if args.max_train_batches is not None and args.max_train_batches > 0:
        max_batches = min(args.max_train_batches, total_batches)
    else:
        max_batches = max(1, int(np.ceil(args.train_frac * total_batches)))

    print(f"Using {max_batches}/{total_batches} batches per epoch (≈ {(max_batches/total_batches):.1%}).")

    for epoch in range(args.epochs):
        start_t = time.time()
        model.train()
        tr_losses, tr_preds, tr_tgts = [], [], []

        for b_idx, batch in enumerate(train_loader):
            if b_idx >= max_batches:
                break  # early stop for fast‑mode
            optimizer.zero_grad()
            if args.model_type == "multimodal":
                outputs = model(batch["image"].to(device), batch["categorical"].to(device), batch["continuous"].to(device))
                labels = batch["label"].to(device)
            elif args.model_type == "image_only":
                outputs = model(batch["image"].to(device))
                labels = batch["label"].to(device)
            else:
                outputs = model(batch["categorical"].to(device), batch["continuous"].to(device))
                labels = batch["label"].to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())
            tr_preds.extend(outputs.argmax(1).detach().cpu().tolist())
            tr_tgts.extend(labels.detach().cpu().tolist())

        train_loss = float(np.mean(tr_losses))
        train_acc = float(accuracy_score(tr_tgts, tr_preds))

        # ---- validation ----
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, args)
        scheduler.step(val_loss)

        epoch_t = time.time() - start_t
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(epoch_t)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | {epoch_t:.1f}s | lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}"
        )

        # ---- save best checkpoint ----
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1:02d}.pt")
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))  # alias
            best_path = ckpt_path
            print(f"    ↳ saved new best model: {ckpt_path}")

    plot_training_history(history, args.output_dir, args.model_type)
    return history, best_path

# -----------------------------------------------------------------------------
# Optional: TabTransformer stand‑alone fine‑tune
# -----------------------------------------------------------------------------
# from train_tab import tune_tab_transformer_model  # keep import at bottom to avoid circulars

# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------

def main() -> Optional[Tuple]:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- optional TabTransformer fine‑tune ----
    if args.tune_tab:
        tune_tab_transformer_model(args, device)
        return

    # ---- dataset ----
    img_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_sources = {
        args.anno_ori: args.imgs_ori,
        args.anno_new: args.imgs_new,
    }
    print("Loading dataset …")
    dataset = MultimodalAMDDataset(data_sources=data_sources, transforms=img_tfms)
    print(f"Total samples: {len(dataset)} | Classes: {dataset.get_num_classes()}")

    # ---- volume‑level split ----
    vols = dataset.df["volume_id"].unique()
    train_vols, val_vols = train_test_split(
        vols,
        test_size=args.val_size,
        stratify=[dataset.get_volume_label(v) for v in vols],
        random_state=args.seed,
    )

    train_idx = dataset.df[dataset.df.volume_id.isin(train_vols)].index.tolist()
    val_idx   = dataset.df[dataset.df.volume_id.isin(val_vols)].index.tolist()

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train vols: {len(train_vols)} | Val vols: {len(val_vols)}")

    # ---- model ----
    model = create_model(args, dataset).to(device)

    # ---- train ----
    print("Starting training …")
    history, best_ckpt = train_model(args, model, train_loader, val_loader, device)
    print(f"Finished. Best val acc: {max(history['val_acc']):.4f} | Best ckpt: {best_ckpt}\n")

    # Return for interactive/IPython use
    return model, history, best_ckpt


if __name__ == "__main__":
    main()
