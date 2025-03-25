import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from model import MultiModalFusionBERT
from dataset import MultiModalDataset
from torch.utils.data import DataLoader

def evaluate(model, device, test_loader):
    model.eval()
    total_loss, correct, total_auc = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, tabular, labels in test_loader:
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            auc = roc_auc_score(labels.cpu().numpy(), torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            total_auc += auc
    accuracy = correct / len(test_loader.dataset)
    auc_score = total_auc / len(test_loader)
    return accuracy, auc_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFusionBERT(image_feature_dim=256, tabular_feature_dim=256, hidden_dim=768, num_heads=8, num_classes=2).to(device)
    model.load_state_dict(torch.load('best_model.pth'))  # Load the trained model

    test_dataset = MultiModalDataset(test_img, test_tab, test_label, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    accuracy, auc_score = evaluate(model, device, test_loader)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test AUC Score: {auc_score:.4f}')