import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ======================== Dataset ========================
class MultiModalEmbeddingDataset(Dataset):
    def __init__(self, image_embeddings, text_embeddings, labels):
        self.image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        self.text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx], self.labels[idx]

# =================== Cross-Attention Module ===================
class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attn(query=query, key=key, value=value)
        return self.layer_norm(query + attn_output)

# =================== Main Model ===================
class MultiModalFusionBERT(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, hidden_dim=768, num_heads=8, num_classes=6, finetune_last_bert_layer=False):
        super().__init__()
        self.image_fc = nn.Linear(image_feature_dim, hidden_dim)
        self.text_fc = nn.Linear(text_feature_dim, hidden_dim)
        self.cross_attn_fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads)
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')

        if finetune_last_bert_layer:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
            for param in self.bert_encoder.encoder.layer[-1].parameters():
                param.requires_grad = True
            if hasattr(self.bert_encoder, 'pooler'):
                for param in self.bert_encoder.pooler.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_features, text_features):
        img_embed = self.image_fc(image_features).unsqueeze(1)
        text_embed = self.text_fc(text_features).unsqueeze(1)
        fused = self.cross_attn_fusion(query=text_embed, key=img_embed, value=img_embed)
        combined_seq = torch.cat([fused, text_embed], dim=1)
        attention_mask = torch.ones(combined_seq.size()[:2], dtype=torch.long, device=combined_seq.device)
        bert_output = self.bert_encoder(inputs_embeds=combined_seq, attention_mask=attention_mask)
        cls_token = bert_output.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)

# =================== Train Function ===================
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for image_feats, text_feats, labels in dataloader:
        image_feats, text_feats, labels = image_feats.to(device), text_feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(image_feats, text_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# =================== Validation Function ===================
def evaluate_model(model, dataloader, device):
    model.eval()
    true_all, pred_all = [], []
    with torch.no_grad():
        for image_feats, text_feats, labels in dataloader:
            image_feats, text_feats = image_feats.to(device), text_feats.to(device)
            outputs = model(image_feats, text_feats)
            preds = outputs.argmax(dim=1).cpu().numpy()
            true_all.extend(labels.numpy())
            pred_all.extend(preds)
    print(classification_report(true_all, pred_all, target_names=[
        "Early AMD", "GA", "Int AMD", "Not AMD", "Scar", "Wet"
    ]))

# =================== Main ===================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your data
    image_embeddings = np.load("D:/AI_Project_BME/Multimodal-Classification/outputs/volume_features.npy")
    text_embeddings = np.load("D:/AI_Project_BME/Multimodal-Classification/outputs/text_embeddings.npy")
    labels = np.load("D:/AI_Project_BME/Multimodal-Classification/outputs/volume_labels.npy")

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Train/val split
    train_img, val_img, train_text, val_text, train_label, val_label = train_test_split(
        image_embeddings, text_embeddings, labels, test_size=0.2, random_state=42
    )

    # Datasets and loaders
    train_dataset = MultiModalEmbeddingDataset(train_img, train_text, train_label)
    val_dataset = MultiModalEmbeddingDataset(val_img, val_text, val_label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, loss, optimizer
    model = MultiModalFusionBERT(image_feature_dim=1024, text_feature_dim=166, num_classes=6,
                                 finetune_last_bert_layer=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Train
    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        evaluate_model(model, val_loader, device)