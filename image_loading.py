######## Modify "root_directory" first before running the code #######

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def find_b_scans_directory(root_path):
    """
    Recursively searches for the 'B-scans' directory.
    Since each level contains only one subdirectory, traverse through until reach the 'B-scans' folder.
    
    :param root_path: The starting directory path (e.g., a specific scan date folder).
    :return: Full path to the 'B-scans' directory or None if not found.
    """
    while True:
        subdirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        if len(subdirs) == 1:  # If there's only one subdirectory, continue traversing
            root_path = os.path.join(root_path, subdirs[0])
        elif "B-scans" in subdirs:
            return os.path.join(root_path, "B-scans")
        else:
            return None  # Unexpected structure, return None

class OCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom PyTorch Dataset for loading OCT images from a hierarchical directory structure.
        
        :param root_dir: Root directory containing patient folders (e.g., 'cirrus_OCT_Imaging_Data').
        :param transform: Optional image transformations for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate over all patient folders
        for patient_id in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue  # Skip non-directory files

            # Iterate over left ('L') and right ('R') eye folders
            for eye in ["L", "R"]:
                eye_path = os.path.join(patient_path, eye)
                if not os.path.isdir(eye_path):
                    continue
                
                # Iterate over different scan dates
                for scan_date in os.listdir(eye_path):
                    scan_date_path = os.path.join(eye_path, scan_date)
                    if not os.path.isdir(scan_date_path):
                        continue

                    # Recursively find the 'B-scans' directory
                    b_scans_path = find_b_scans_directory(scan_date_path)
                    if b_scans_path and os.path.isdir(b_scans_path):
                        for img_name in os.listdir(b_scans_path):
                            img_path = os.path.join(b_scans_path, img_name)
                            if img_path.endswith(".jpg") or img_path.endswith(".png"):  # Process only image files
                                self.image_paths.append(img_path)
                                self.labels.append((patient_id, eye, scan_date))  # Store patient ID, eye, scan date

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label.
        
        :param idx: Index of the image to fetch.
        :return: Tuple (image, label) where:
                 - image is a transformed tensor.
                 - label is a tuple (patient_id, eye, scan_date).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and convert to grayscale
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)

        return image, label

# Transformation pipeline for data augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256 pixels
    transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Initialize the dataset
root_directory = "cirrus_OCT_Imaging_Data"  # Need to Modify this path accordingly
dataset = OCTDataset(root_dir=root_directory, transform=transform)

# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Note
# for images, labels in dataloader:
#     print(images.shape)  # Expected shape: (batch_size, 1, 256, 256)
#     print(labels)  # Example: [('00003162', 'L', '20080818'), ('00004567', 'R', '20150609'), ...]
#     break  # Display only the first batch
