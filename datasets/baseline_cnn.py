"""
Author: Son Phat Tran
This file defines the PyTorch image dataset used for training the baseline CNN model
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_datasets(data_dir):
    """
    Create train and validation datasets using ImageFolder

    Args:
        data_dir (str): Path to the data directory
    """
    # Define transforms for training data
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Define transforms for validation data
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=val_transform
    )

    return train_dataset, val_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """
    Create train and validation data loaders

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of worker processes
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage
def setup_data(data_dir, batch_size=32, num_workers=4):
    """
    Complete setup of datasets and data loaders

    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of worker processes
    """
    # Create datasets
    train_dataset, val_dataset = create_datasets(data_dir)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size,
        num_workers
    )

    # Get the number of classes
    num_classes = len(train_dataset.classes)

    # Print dataset information
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    # Create two data loader for train and test set
    train_loadr, val_loadr, total_classes = setup_data(
        data_dir="raw_dataset/small",
        batch_size=32,
        num_workers=4
    )
