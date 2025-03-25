import torch
from sklearn.metrics import roc_auc_score

def train_model(model, image_encoder, tabular_encoder, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total_auc = 0, 0, 0
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
        total_auc += roc_auc_score(labels.cpu().numpy(), torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return total_loss / len(train_loader), correct / len(train_loader.dataset), total_auc / len(train_loader)