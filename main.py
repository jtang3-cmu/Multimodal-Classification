import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset
import timm  # Used for loading RETFound
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image

# ===============================
# 1. Image Model (RETFound Encoder)
# ===============================
class RETFoundEncoder(nn.Module):
    def __init__(self, model_name="timm/retfound"):
        super(RETFoundEncoder, self).__init__()
        # Load the RETFound model and remove the classification head
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feature_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)  # Return image features

# ===============================
# 2. Tabular Data Model (TabTransformer)
# ===============================
class TabTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TabTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ===============================
# 3. Cross-Attention Fusion Module
# ===============================
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        # Use PyTorch's MultiheadAttention to implement cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        """
        query: [batch, seq_len, hidden_dim] - e.g., tabular features
        key, value: [batch, seq_len, hidden_dim] - e.g., image features
        """
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        fused = self.layer_norm(query + attn_output)
        return fused

# ===============================
# 4. Multi-Modal Model: First perform cross-attention fusion,
#    then use a pre-trained/fine-tuned BERT as the decoder
# ===============================
class MultiModalFusionBERT(nn.Module):
    def __init__(self, image_feature_dim, tabular_feature_dim, hidden_dim=768, num_heads=8, num_classes=2):
        super(MultiModalFusionBERT, self).__init__()
        # Map features from each modality to the same dimension
        self.image_fc = nn.Linear(image_feature_dim, hidden_dim)
        self.tabular_fc = nn.Linear(tabular_feature_dim, hidden_dim)
        
        # Cross-attention fusion module: using tabular features as query and image features as key/value
        self.cross_attn_fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        
        # Use a pre-trained BERT as the decoder (load pre-trained weights)
        self.bert_decoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Classifier head: based on the [CLS] token output from BERT decoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image_features, tabular_features):
        # 1. Map both modalities to the same hidden_dim
        img_embed = self.image_fc(image_features)   # [batch, hidden_dim]
        tab_embed = self.tabular_fc(tabular_features) # [batch, hidden_dim]
        
        # 2. Expand vectors to sequences: treat each as a single token
        # Here, the tabular token acts as query, and the image token acts as key/value
        tab_seq = tab_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        img_seq = img_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 3. Fuse features via cross-attention: let tabular info attend to image info
        fused_tab = self.cross_attn_fusion(query=tab_seq, key=img_seq, value=img_seq)  # [batch, 1, hidden_dim]
        
        # 4. Construct an input sequence for the BERT decoder (e.g., concatenate the fused token with the original tabular token)
        combined_seq = torch.cat([fused_tab, tab_seq], dim=1)  # [batch, 2, hidden_dim]
        batch_size, seq_length, _ = combined_seq.size()
        attention_mask = torch.ones(batch_size, seq_length, device=combined_seq.device, dtype=torch.long)
        
        # 5. Pass through the pre-trained BERT decoder
        bert_outputs = self.bert_decoder(inputs_embeds=combined_seq, attention_mask=attention_mask)
        # Take the first token's output as the global [CLS] representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_dim]
        
        # 6. Classification
        logits = self.classifier(cls_output)
        return logits

# ===============================
# 5. Custom Dataset Class
# ===============================
class MultiModalDataset(Dataset):
    def __init__(self, image_paths, tabular_data, labels, transform=None):
        self.image_paths = image_paths
        self.tabular_data = tabular_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and convert to RGB
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Load tabular data
        tabular_features = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, tabular_features, label

# ===============================
# 6. Training Function
# ===============================
def train_model(model, image_encoder, tabular_encoder, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for images, tabular, labels in train_loader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
        optimizer.zero_grad()
        # First, extract image and tabular features using RETFoundEncoder and TabTransformer
        with torch.no_grad():
            img_features = image_encoder(images)  # [batch, image_feature_dim]
            tab_features = tabular_encoder(tabular) # [batch, tabular_feature_dim]
        outputs = model(img_features, tab_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)

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
        train_loss, train_acc = train_model(model, image_encoder, tabular_encoder, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
