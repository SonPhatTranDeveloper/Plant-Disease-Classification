"""
Author: Son Phat Tran
This file contains the code for BLIP fine-tuning using a classification head
"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BlipModel

from utils.data import load_image_label_pairs
from datasets.clip_classification_head import ImageLabelDataset, CROP_AND_ROTATION_TRANSFORM, NO_TRANSFORM


class BLIPClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="Salesforce/blip-image-captioning-large"):
        super().__init__()

        # Load pretrained CLIP model
        self.blip = BlipModel.from_pretrained(pretrained_model)

        # Freeze CLIP parameters
        for param in self.blip.parameters():
            param.requires_grad = False

        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 384),  # Reduced from input dim 768
            nn.LayerNorm(384),  # Added LayerNorm for better stability
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout for larger model
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes)
        )

    def forward(self, pixel_values):
        # Get vision encoder outputs
        vision_outputs = self.blip.vision_model(pixel_values)
        pooled_output = vision_outputs.pooler_output

        # Pass through classification head
        logits = self.classifier(pooled_output)
        return logits


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on given data loader and return top-1 and top-5 accuracies
    """
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get top-5 predictions
            _, pred_5 = outputs.topk(5, 1, True, True)
            pred_5 = pred_5.t()
            correct = pred_5.eq(labels.view(1, -1).expand_as(pred_5))

            # Top-1 accuracy
            correct_1 += correct[0].sum().item()

            # Top-5 accuracy
            correct_5 += correct.any(dim=0).sum().item()

            total += labels.size(0)

            # Store predictions and labels for confusion matrix
            all_predictions.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_acc = 100. * correct_1 / total
    top5_acc = 100. * correct_5 / total

    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=5e-5,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1
    )

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{train_loss / len(train_loader):.3f}'
            })

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        print(f'Validation Top-1 Accuracy: {val_metrics["top1_accuracy"]:.2f}%')
        print(f'Validation Top-5 Accuracy: {val_metrics["top5_accuracy"]:.2f}%')

        # Save best model based on top-1 accuracy
        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1_acc': val_metrics["top1_accuracy"],
                'val_top5_acc': val_metrics["top5_accuracy"],
            }, 'weights/best_blip_classification_head.pth')


if __name__ == "__main__":
    # Load train and test set
    train_pairs = load_image_label_pairs("raw_dataset/small/train", convert_label=lambda x: x)
    train_paths = [item[0] for item in train_pairs]
    train_labels = [item[1] for item in train_pairs]

    test_pairs = load_image_label_pairs("raw_dataset/small/test", convert_label=lambda x: x)
    test_paths = [item[0] for item in test_pairs]
    test_labels = [item[1] for item in test_pairs]

    # Create dataset and dataloader
    train_dataset = ImageLabelDataset(
        image_paths=train_paths,
        labels=train_labels,
        num_augmentations=19,
        transform=CROP_AND_ROTATION_TRANSFORM
    )

    test_dataset = ImageLabelDataset(
        image_paths=test_paths,
        labels=test_labels,
        num_augmentations=0,
        transform=NO_TRANSFORM
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 15

    # Total classes
    total_classes = len(set(train_labels))

    # Initialize model
    model = BLIPClassifier(num_classes=total_classes)
    model = model.to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs, device)