"""
Author: Son Phat Tran
This file initiates the training of the baseline CNN model
"""
import torch

from datasets.baseline_cnn import setup_data
from models.baseline_cnn import initialize_training, train_model

if __name__ == "__main__":
    # Create two data loader for train and test set
    train_loader, val_loader, num_classes = setup_data(
        data_dir="raw_dataset/small",
        batch_size=32,
        num_workers=4
    )

    # Initialize training and train the model
    model, criterion, optimizer, device = initialize_training(
        num_classes=num_classes,
        learning_rate=0.001
    )

    train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


