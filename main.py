import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import MultimodalAMDDataset
from models import create_model
from training import train_model, test_model
from utils import set_seed
def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal AMD Classification')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='D:/AI_Project_BME/annotation_modified_final_forTrain.xlsx', 
                        help='Path to the tabular data file')
    parser.add_argument('--image_dir', type=str, default='D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data', 
                        help='Path to the image directory')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save models and results')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='multimodal', 
                        choices=['multimodal', 'image_only', 'tabular_only'],
                        help='Type of model to train')
    parser.add_argument('--freeze_encoders', action='store_true', 
                        help='Freeze encoder weights')
    parser.add_argument('--tab_dim', type=int, default=64, 
                        help='Dimension of tabular features')
    parser.add_argument('--hidden_dim', type=int, default=768, 
                        help='Hidden dimension for fusion')
    parser.add_argument('--num_heads', type=int, default=8, 
                        help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.2, 
                        help='Validation set size ratio (from training set)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    return parser.parse_args()

def main():
    # Parse arguments using the existing function
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"Loading dataset from {args.data_path} and {args.image_dir}")
    dataset = MultimodalAMDDataset(
        tabular_path=args.data_path,
        image_root_dir=args.image_dir,
        transforms=image_transforms
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Split dataset into train, validation, and test sets
    
    indices = list(range(len(dataset)))
    
    # First split off the test set
    train_val_indices, test_indices = train_test_split(
        indices, 
        test_size=args.test_size, 
        stratify=[dataset[i]['label'].item() for i in indices],
        random_state=args.seed
    )
    
    # Then split the remaining data into train and validation sets
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=args.val_size / (1 - args.test_size),  # Adjust for the remaining percentage
        stratify=[dataset[i]['label'].item() for i in train_val_indices],
        random_state=args.seed
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(args, dataset)
    model.to(device)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    history, final_model_path = train_model(args, model, train_loader, val_loader, device)
    
    # Load final model for testing
    checkpoint = torch.load(final_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded final model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.4f}")
    
    # Test model
    print("Evaluating model on test set...")
    test_loss, test_acc = test_model(args, model, test_loader, device, dataset)
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"All results saved to {args.output_dir}")
    
    return model, history, test_acc

if __name__ == "__main__":
    main()
