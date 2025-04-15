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
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal AMD Classification')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='D:/AI_Project_BME/annotation_modified_final_forTrain.xlsx', 
                        help='Path to the tabular data file')
    parser.add_argument('--image_dir', type=str, default='D:/cleaning_GUI_annotated_Data/Cirrus_OCT_Imaging_Data', 
                        help='Path to the image directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/', 
                        help='Directory to save models and results')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='multimodal', 
                        choices=['multimodal', 'image_only', 'tabular_only'],
                        help='Type of model to train')
    parser.add_argument('--image_encoder_type', type=str, default='resnet50',
                        choices=['resnet50', 'retfound'],
                        help='Type of image encoder to use')
    parser.add_argument('--retfound_weights', type=str, 
                        default='RETFound_MAE/RETFound_mae_natureOCT.pth',
                        help='Path to RETFound pretrained weights')
    parser.add_argument('--freeze_encoders', action='store_true', 
                        help='Freeze encoder weights')
    parser.add_argument('--tab_dim', type=int, default=64, 
                        help='Dimension of tabular features')
    parser.add_argument('--hidden_dim', type=int, default=1024, 
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
    
    # New arguments for TabTransformer tuning on tabular data
    parser.add_argument('--tune_tab', action='store_true', 
                        help='Run fine-tuning of TabTransformer on tabular data')
    parser.add_argument('--tab_data_path', type=str, default='annotation_modified_final_forTrain_v2.xlsx', 
                        help='Path to tabular data file for TabTransformer tuning')
    parser.add_argument('--tab_epochs', type=int, default=20, 
                        help='Number of epochs for TabTransformer tuning')
    parser.add_argument('--tab_batch_size', type=int, default=32, 
                        help='Batch size for TabTransformer tuning')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If fine-tuning TabTransformer on tabular data is requested, run that routine and exit
    if args.tune_tab:
        from train import tune_tab_transformer_model
        tune_tab_transformer_model(args, device)
        return
    
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
