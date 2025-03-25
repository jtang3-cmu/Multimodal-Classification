import torch
from PIL import Image
from torch.utils.data import Dataset

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