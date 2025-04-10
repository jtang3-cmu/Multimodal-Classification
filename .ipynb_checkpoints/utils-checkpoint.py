import matplotlib.pyplot as plt
import os
import numpy as np
import random
import torch

def save_datasets(train_dataset, val_dataset, test_dataset, output_dir):
    """Save train, validation, and test datasets to disk.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        output_dir: Directory to save the datasets
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train_dataset.pt')
    val_path = os.path.join(output_dir, 'val_dataset.pt')
    test_path = os.path.join(output_dir, 'test_dataset.pt')

    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    torch.save(test_dataset, test_path)

    print(f"Datasets saved to {output_dir}")
    return train_path, val_path, test_path

def load_datasets(output_dir):
    """Load train, validation, and test datasets from disk.
    
    Args:
        output_dir: Directory containing the saved datasets
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_path = os.path.join(output_dir, 'train_dataset.pt')
    val_path = os.path.join(output_dir, 'val_dataset.pt')
    test_path = os.path.join(output_dir, 'test_dataset.pt')
    
    # Check if all dataset files exist
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        raise FileNotFoundError("One or more dataset files not found in the specified directory")
    
    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)
    test_dataset = torch.load(test_path)
    
    print(f"Datasets loaded from {output_dir}")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset
    
def plot_training_history(history, output_dir=None, model_type=None):
    """
    Plot training and validation metrics from training history.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save the plot
        model_type: Type of model (for naming the saved file)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.semilogy(history['lr'], label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot epoch times
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(history['epoch_times'])+1), history['epoch_times'])
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output directory is provided
    if output_dir and model_type:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_training_history.png'))

def set_seed(seed):
    """
    Set random seed for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")