import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MultiModalFusionBERT, RETFoundEncoder, TabTransformer
from dataset import MultiModalDataset
from train import train_model

# ===============================
# 7. Main Program
# ===============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load tabular data (assumed to be saved in a CSV file)
    df = pd.read_csv("clinical_data.csv")
    image_paths = df["image_path"].values
    tabular_data = df.drop(["image_path", "label"], axis=1).values
    labels = df["label"].values

    # Standardize tabular data
    scaler = StandardScaler()
    tabular_data = scaler.fit_transform(tabular_data)

    # Split into training and validation sets
    train_img, val_img, train_tab, val_tab, train_label, val_label = train_test_split(
        image_paths, tabular_data, labels, test_size=0.2, random_state=42
    )

    train_dataset = MultiModalDataset(train_img, train_tab, train_label, transform)
    val_dataset = MultiModalDataset(val_img, val_tab, val_label, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize encoders
    image_encoder = RETFoundEncoder().to(device)
    tabular_encoder = TabTransformer(input_dim=train_tab.shape[1], output_dim=256).to(device)

    # Initialize multi-modal model
    model = MultiModalFusionBERT(image_encoder.feature_dim, 256, hidden_dim=768, num_heads=8, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    for epoch in range(10):
        train_loss, train_acc, train_auc = train_model(model, image_encoder, tabular_encoder, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}, AUC = {train_auc:.4f}")
