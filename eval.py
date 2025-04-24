# ===== eval.py =====
"""Standalone evaluation script for inference on the test set.

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
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import MultimodalAMDDataset
from model import create_model

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
# CLI
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluation")
    p.add_argument("--model_path", type=str, required=True, help="Checkpoint to load (.pt)")
    p.add_argument("--output_dir", type=str, default="./eval_outputs", help="Dir to save reports")
    p.add_argument("--model_type", type=str, default="multimodal",
                   choices=["multimodal", "image_only", "tabular_only"])
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

    # Test split (same stratified volume logic as training)
    from sklearn.model_selection import train_test_split
    volume_ids = dataset.df["volume_id"].unique()
    _, test_volumes = train_test_split(
        volume_ids,
        test_size=0.2,
        stratify=[dataset.get_volume_label(v) for v in volume_ids],
        random_state=args.seed,
    )
    test_indices = dataset.df[dataset.df["volume_id"].isin(test_volumes)].index.tolist()
    test_set = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    dummy_args = argparse.Namespace(**{
        "model_type": args.model_type,
        "image_encoder_type": "resnet50",
        "retfound_weights": "",
        "freeze_encoders": False,
        "tab_dim": 64,
        "hidden_dim": 1024,
        "num_heads": 8,
    })
    model = create_model(dummy_args, dataset)
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {args.model_path} (val acc={checkpoint.get('val_acc', 'N/A')})")

    # Inference
    y_true, y_pred = _infer(model, test_loader, device, args.model_type)
    acc = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Reports
    cls_names = dataset.get_label_map()
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=cls_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cls_names, yticklabels=cls_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save predictions to CSV
    import pandas as pd
    df_out = pd.DataFrame({"true": y_true, "pred": y_pred,
                           "true_label": [cls_names[i] for i in y_true],
                           "pred_label": [cls_names[i] for i in y_pred]})
    csv_path = os.path.join(args.output_dir, "predictions.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

if __name__ == "__main__":
    main()
