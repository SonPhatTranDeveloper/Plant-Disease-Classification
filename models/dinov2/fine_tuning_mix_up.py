"""
Author: Son Phat Tran
This code contains the code fine-tuning DINOv2 classification style with mix-up.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

from models.dinov2.layers.linear import create_linear_input, LinearClassifier, ModelWithIntermediateLayers
from datasets.dinov2 import AugmentedDINOv2Dataset, STANDARD_TRANSFORM
from utils.augmentation import mix_up_data
from utils.data import load_image_label_pairs


def train_epoch(model, train_loader, criterion, optimizer, device, cut_mix_prob=0.5):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        # Move inputs to device
        if isinstance(inputs, (list, tuple)):
            inputs = [input.to(device) for input in inputs]
        else:
            inputs = inputs.to(device)

        # Move labels to device
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Mix up images and labels
        mixed_images, labels_a, labels_b, lam = mix_up_data(inputs, labels)

        # Get the outputs
        outputs = model(mixed_images)

        # Calculate cut mix loss
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        # Calculate top-1 and top-5 accuracy
        _, predicted = outputs.topk(5, 1, True, True)
        total += labels.size(0)

        # Top-1 accuracy
        correct_top1 += (predicted[:, 0] == labels).sum().item()

        # Top-5 accuracy
        correct_top5 += sum([1 for i, label in enumerate(labels)
                             if label in predicted[i]])

    epoch_loss = running_loss / len(train_loader)
    epoch_acc_top1 = 100 * correct_top1 / total
    epoch_acc_top5 = 100 * correct_top5 / total

    return epoch_loss, epoch_acc_top1, epoch_acc_top5


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            # Move inputs to device
            if isinstance(inputs, (list, tuple)):
                inputs = [input.to(device) for input in inputs]
            else:
                inputs = inputs.to(device)

            # Move labels to device
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()

            # Calculate top-1 and top-5 accuracy
            _, predicted = outputs.topk(5, 1, True, True)
            total += labels.size(0)

            # Top-1 accuracy
            correct_top1 += (predicted[:, 0] == labels).sum().item()

            # Top-5 accuracy
            correct_top5 += sum([1 for i, label in enumerate(labels)
                                 if label in predicted[i]])

    val_loss = running_loss / len(val_loader)
    val_acc_top1 = 100 * correct_top1 / total
    val_acc_top5 = 100 * correct_top5 / total

    return val_loss, val_acc_top1, val_acc_top5


class Dino(torch.nn.Module):
    def __init__(self, type, device, num_classes):
        super().__init__()
        # get feature model
        model = torch.hub.load(
            "facebookresearch/dinov2", type, pretrained=True
        ).to(device)
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )
        self.feature_model = ModelWithIntermediateLayers(
            model, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).to(device)

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224).to(device)
            sample_output = self.feature_model(sample_input)

        # get linear readout
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]
        self.classifier = LinearClassifier(
            out_dim, use_n_blocks=1, use_avgpool=True, num_classes=num_classes
        ).to(device)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Create training and validation datasets
    train_pairs = load_image_label_pairs("raw_dataset/small/train", convert_label=lambda x: x)
    train_paths = [item[0] for item in train_pairs]
    train_labels = [item[1] for item in train_pairs]

    test_pairs = load_image_label_pairs("raw_dataset/small/test", convert_label=lambda x: x)
    test_paths = [item[0] for item in test_pairs]
    test_labels = [item[1] for item in test_pairs]

    # Get the total classes
    total_classes = len(set(train_labels))

    # Create the train and test dataloader
    train_dataset = AugmentedDINOv2Dataset(
        image_paths=train_paths,
        labels=train_labels,
        num_augmentations=19
    )

    test_dataset = AugmentedDINOv2Dataset(
        image_paths=test_paths,
        labels=test_labels,
        num_augmentations=0,
        transform=STANDARD_TRANSFORM
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Dino("dinov2_vits14_reg", device, total_classes)

    # Disable gradient for feature model
    for param in model.feature_model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    num_epochs = 15
    best_acc = 0.0

    print("Starting training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_acc_top1, train_acc_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on validation set
        val_loss, val_acc_top1, val_acc_top5 = evaluate(
            model, val_loader, criterion, device
        )

        # Step the scheduler
        exp_lr_scheduler.step()

        # Print epoch statistics
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy - Top-1: {train_acc_top1:.2f}%, Top-5: {train_acc_top5:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy - Top-1: {val_acc_top1:.2f}%, Top-5: {val_acc_top5:.2f}%")

        # Save best model
        if val_acc_top1 > best_acc:
            best_acc = val_acc_top1
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model")

    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
