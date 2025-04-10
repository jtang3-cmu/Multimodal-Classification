import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import MultimodalAMDDataset
from model import create_model
from train import train_model, test_model
from utils import set_seed, save_datasets, load_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal AMD Classification')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='D:/AI_Project_BME/annotation_modified_final_forTrain.xlsx', 
                        help='Path to the tabular data file')
    parser.add_argument('--image_dir', type=str, default='D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data', 
                        help='Path to the image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/', 
                        help='Directory to save models and results')
    
    # Add dataset saving/loading parameters
    parser.add_argument('--dataset_dir', type=str, default='./datasets',
                       help='Directory to save/load datasets')
    parser.add_argument('--load_datasets', action='store_true',
                       help='Load pre-processed datasets instead of creating new ones')
    parser.add_argument('--save_datasets', action='store_true',
                       help='Save datasets after train-test split')
    
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
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.2, 
                        help='Validation set size ratio (from training set)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load or create datasets
    if args.load_datasets:
        # Load pre-processed datasets
        train_dataset, val_dataset, test_dataset = load_datasets(args.dataset_dir)
    else:
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
        
        # Get unique volume IDs
        volume_ids = dataset.df['volume_id'].unique()
        
        # Split at the volume level first
        train_val_volumes, test_volumes = train_test_split(
            volume_ids,
            test_size=args.test_size,
            stratify=[dataset.get_volume_label(vid) for vid in volume_ids],
            random_state=args.seed
        )
        
        # Then split train_val into train and validation
        train_volumes, val_volumes = train_test_split(
            train_val_volumes,
            test_size=args.val_size / (1 - args.test_size),
            stratify=[dataset.get_volume_label(vid) for vid in train_val_volumes],
            random_state=args.seed
        )
        
        # Now get the indices for each split
        train_indices = dataset.df[dataset.df['volume_id'].isin(train_volumes)].index.tolist()
        val_indices = dataset.df[dataset.df['volume_id'].isin(val_volumes)].index.tolist()
        test_indices = dataset.df[dataset.df['volume_id'].isin(test_volumes)].index.tolist()

        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"Train set: {len(train_volumes)} volumes, {len(train_indices)} B-scans")
        print(f"Validation set: {len(val_volumes)} volumes, {len(val_indices)} B-scans")
        print(f"Test set: {len(test_volumes)} volumes, {len(test_indices)} B-scans")

        
        # Save datasets if requested
        if args.save_datasets:
            save_datasets(train_dataset, val_dataset, test_dataset, args.dataset_dir)
    
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
    checkpoint = torch.load(final_model_path, weights_only=False)
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
