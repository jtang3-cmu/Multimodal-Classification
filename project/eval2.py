# ===== eval.py =====
"""Standalone evaluation script for inference on the test set and entire dataset.

Example usage
-------------
$ python eval.py \
    --model_path ./outputs/best_model.pt \
    --output_dir ./outputs/eval \
    --batch_size 32
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import MultimodalAMDDataset
from model import create_model

# --------------------------------------------------
# Helper: run inference on a loader and return predictions with volume IDs
# --------------------------------------------------

def _infer_with_ids(model, loader, device, model_type, dataset):
    model.eval()
    preds, targets, volume_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == "multimodal":
                categorical = batch["categorical"].to(device)
                continuous = batch["continuous"].to(device)
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images, categorical, continuous)
            elif model_type == "image_only":
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
            else:
                categorical = batch["categorical"].to(device)
                continuous = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(categorical, continuous)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            current_volume_ids = [dataset.df.iloc[i]['volume_id'] for i in batch['index']]
            volume_ids.extend(current_volume_ids)
    return np.array(targets), np.array(preds), volume_ids

# --------------------------------------------------
# Helper: run inference on a loader
# --------------------------------------------------

def _infer(model, loader, device, model_type):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == "multimodal":
                categorical = batch["categorical"].to(device)
                continuous = batch["continuous"].to(device)
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images, categorical, continuous)
            elif model_type == "image_only":
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
            else:
                categorical = batch["categorical"].to(device)
                continuous = batch["continuous"].to(device)
                labels = batch["label"].to(device)
                outputs = model(categorical, continuous)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.array(targets), np.array(preds)

# --------------------------------------------------
# Function to generate and save confusion matrix
# --------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"{title} saved to {output_path}")

# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluation")
    p.add_argument("--model_path", type=str, required=True, help="Checkpoint to load (.pt)")
    p.add_argument("--output_dir", type=str, default="./eval_outputs", help="Dir to save reports")
    p.add_argument("--model_type", type=str, default="multimodal",
                     choices=["multimodal", "image_only", "tabular_only"])
    p.add_argument("--image_encoder_type", type=str, default="retfound", choices=["resnet50", "retfound"], help="CNN encoder")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data — replicate transforms from training
    img_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_sources = {
        r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_ori.xlsx": r"D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data",
        r"D:/AI_Project_BME/vol-wise_annotations/vol_anno_new.xlsx": r"D:/cleaning_GUI_annotated_Data/New_Data",
    }
    dataset = MultimodalAMDDataset(data_sources=data_sources, transforms=img_t)
    cls_names = dataset.get_label_map()

    # Test split (same stratified volume logic as training)
    volume_ids = dataset.df["volume_id"].unique()
    train_volumes, test_volumes = train_test_split(
        volume_ids,
        test_size=0.2,
        stratify=[dataset.get_volume_label(v) for v in volume_ids],
        random_state=args.seed,
    )
    train_indices = dataset.df[dataset.df["volume_id"].isin(train_volumes)].index.tolist()
    test_indices = dataset.df[dataset.df["volume_id"].isin(test_volumes)].index.tolist()

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    all_set = dataset  # The entire dataset

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_loader = DataLoader(all_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    dummy_args = argparse.Namespace(**{
        "model_type": args.model_type,
        "image_encoder_type": "retfound",
        "retfound_weights": "RETFound_MAE/RETFound_mae_natureOCT.pth",
        "freeze_encoders": False,
        "tab_dim": 64,
        "hidden_dim": 1024,
        "num_heads": 8,
    })
    model = create_model(dummy_args, dataset)
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from {args.model_path} (val acc={checkpoint.get('val_acc', 'N/A')})")

    # Inference and reports for test set
    print("\n--- Test Set Evaluation ---")
    y_true_test, y_pred_test = _infer(model, test_loader, device, args.model_type)
    acc_test = accuracy_score(y_true_test, y_pred_test)
    print(f"Test accuracy: {acc_test:.4f}")
    print("\nTest Classification report:\n", classification_report(y_true_test, y_pred_test, target_names=cls_names))
    _plot_confusion_matrix(y_true_test, y_pred_test, cls_names, os.path.join(args.output_dir, "test_confusion_matrix.png"), title="Test Confusion Matrix")

    # Inference and reports for the entire dataset
    print("\n--- Entire Dataset Evaluation ---")
    y_true_all, y_pred_all = _infer(model, all_loader, device, args.model_type)
    acc_all = accuracy_score(y_true_all, y_pred_all)
    print(f"Entire dataset accuracy: {acc_all:.4f}")
    print("\nEntire Dataset Classification report:\n", classification_report(y_true_all, y_pred_all, target_names=cls_names))
    _plot_confusion_matrix(y_true_all, y_pred_all, cls_names, os.path.join(args.output_dir, "all_confusion_matrix.png"), title="Entire Dataset Confusion Matrix")

    # Create DataFrame for all predictions
    all_results_df = pd.DataFrame({
        "volume_id": test_volume_ids,
        "split": "test",
        "true": y_true_test,
        "predicted": y_pred_test,
        "true_label": [cls_names[i] for i in y_true_test],
        "predicted_label": [cls_names[i] for i in y_pred_test],
    })

    all_results_df_all = pd.DataFrame({
        "volume_id": all_volume_ids,
        "split": "all",
        "true": y_true_all,
        "predicted": y_pred_all,
        "true_label": [cls_names[i] for i in y_true_all],
        "predicted_label": [cls_names[i] for i in y_pred_all],
    })

    all_results_df = pd.concat([all_results_df, all_results_df_all], ignore_index=True)

    # Save all predictions to CSV
    all_results_path = os.path.join(args.output_dir, f'{args.model_type}_all_split_results.csv')
    all_results_df.to_csv(all_results_path, index=False)
    print(f"\nAll predictions saved to {all_results_path}")
    print("\nFirst 5 rows of all predictions:")
    print(all_results_df.head())

if __name__ == "__main__":
    main()