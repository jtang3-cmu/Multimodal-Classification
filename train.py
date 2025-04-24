"""Train script for multimodal AMD classification.

This file now absorbs all CLI/data‑loading logic that previously lived in
*main.py*.  It **only performs training** (with on‑the‑fly validation).  After
each epoch it saves a checkpoint if the validation accuracy improves and keeps
`best_model.pt` symlink/dup pointing to the best epoch so far.  No test‑set
inference is performed here – use **eval.py** instead.
"""

import os
import argparse
import time
from typing import Tuple, Dict, List

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

# ----------------------------
# Argument parsing
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal AMD Training – train only; use eval.py for inference")

    # I/O
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save checkpoints & plots")

    # Model
    parser.add_argument("--model_type", type=str, default="multimodal", choices=["multimodal", "image_only", "tabular_only"], help="Backbone variant")
    parser.add_argument("--image_encoder_type", type=str, default="resnet50", choices=["resnet50", "retfound"], help="CNN encoder")
    parser.add_argument("--retfound_weights", type=str, default="RETFound_MAE/RETFound_mae_natureOCT.pth", help="Path to RETFound weights")
    parser.add_argument("--freeze_encoders", action="store_true", help="Freeze encoders during training")
    parser.add_argument("--tab_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)

    # Training hyper‑params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of data (at volume level) for validation")
    parser.add_argument("--seed", type=int, default=42)

    # Data sources hard‑coded here for now – you can expose as args if desired
    parser.add_argument("--anno_ori", type=str, default=r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_ori.xlsx")
    parser.add_argument("--imgs_ori", type=str, default=r"D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data")
    parser.add_argument("--anno_new", type=str, default=r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_new.xlsx")
    parser.add_argument("--imgs_new", type=str, default=r"D:/cleaning_GUI_annotated_Data/New_Data")

    # Optional: fine‑tuning TabTransformer only
    parser.add_argument("--tune_tab", action="store_true", help="Fine‑tune TabTransformer only and exit")
    parser.add_argument("--tab_data_path", type=str, default="annotation_modified_final_forTrain_v3.xlsx")
    parser.add_argument("--tab_epochs", type=int, default=20)
    parser.add_argument("--tab_batch_size", type=int, default=32)

    return parser.parse_args()

# ----------------------------
# Utility – single‑batch evaluate (val only)
# ----------------------------

def _evaluate(model: torch.nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, args) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    preds: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for batch in loader:
            if args.model_type == "multimodal":
                images = batch["image"].to(device)
                cat = batch["categorical"].to(device)
                cont = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images, cat, cont)
            elif args.model_type == "image_only":
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
            else:  # tabular_only
                cat = batch["categorical"].to(device)
                cont = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(cat, cont)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return float(np.mean(losses)), float(accuracy_score(targets, preds))

# ----------------------------
# Core training loop (with best‑checkpoint save)
# ----------------------------

def train_model(args, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Tuple[Dict[str, List[float]], str]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, verbose=True)

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"]}
    best_acc = -1.0
    best_path = None

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_preds: List[int] = []
        tr_tgts: List[int] = []

        for batch in train_loader:
            optimizer.zero_grad()
            if args.model_type == "multimodal":
                images = batch["image"].to(device)
                cat = batch["categorical"].to(device)
                cont = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images, cat, cont)
            elif args.model_type == "image_only":
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
            else:
                cat = batch["categorical"].to(device)
                cont = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(cat, cont)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())
            tr_preds.extend(outputs.argmax(1).detach().cpu().numpy())
            tr_tgts.extend(labels.detach().cpu().numpy())

        train_loss = float(np.mean(tr_losses))
        train_acc = float(accuracy_score(tr_tgts, tr_preds))

        # -------------------- validation --------------------
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device, args)
        scheduler.step(val_loss)

        # -------------------- bookkeeping -------------------
        epoch_time = time.time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(epoch_time)

        print(f"Epoch {epoch+1:03d}/{args.epochs} | time {epoch_time:.1f}s | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

        # save checkpoint if best
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
            # keep/overwrite alias
            best_alias = os.path.join(args.output_dir, "best_model.pt")
            torch.save(ckpt, best_alias)
            best_path = ckpt_path
            print(f"  >>> saved new best model to {ckpt_path}")

    # after training finish → plot history
    plot_training_history(history, args.output_dir, args.model_type)
    return history, (best_path or "")

# ----------------------------
# Optional: TabTransformer fine‑tune only
# (kept intact from previous version)
# ----------------------------
from train_tab import tune_tab_transformer_model

# ----------------------------
# main()
# ----------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- optional stand‑alone TabTransformer fine‑tuning ----
    if args.tune_tab:
        tune_tab_transformer_model(args, device)
        return

    # ---- build dataset ----
    image_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_sources = {
        args.anno_ori: args.imgs_ori,
        args.anno_new: args.imgs_new,
    }
    print("Loading dataset …")
    dataset = MultimodalAMDDataset(data_sources=data_sources, image_transforms=image_tfms)
    print(f"Dataset size: {len(dataset)} – classes: {dataset.get_num_classes()}")

    # ---- volume‑level split into train/val ----
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model ----
    model = create_model(args, dataset).to(device)

    # ---- training ----
    print(f"Starting training for {args.epochs} epochs …")
    history, best_ckpt = train_model(args, model, train_loader, val_loader, device)
    print(f"Training done. Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Best checkpoint: {best_ckpt}")

if __name__ == "__main__":
    main()