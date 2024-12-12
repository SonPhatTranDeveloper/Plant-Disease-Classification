"""
Author: Son Phat Tran
This code contains the code fine-tuning DINOv2, classification style.
"""
import os
from typing import List, Union, Tuple

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from functools import partial
from tqdm import tqdm


class ResizeAndPad:
    def __init__(self, target_size, multiple):
        """
        Helper class to perform resize and padding on the image
        """
        self.target_size = target_size
        self.multiple = multiple

    def __call__(self, img):
        """
        Call transformation on the image
        """
        # Resize the image
        img = transforms.Resize(self.target_size, interpolation=InterpolationMode.BICUBIC)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = transforms.Pad(
            (pad_width // 2,
             pad_height // 2,
             pad_width - pad_width // 2,
             pad_height - pad_height // 2)
        )(img)

        return img


def load_image_label_pairs(dataset_path: str, convert_label) -> List[Tuple[str, str]]:
    """
    Load the oneshot dataset classification (image + label) from a folder
    Note that the folder must have the following structure:
    - class 1:
        - Image 1.png
        - Image 2.png
    - class 2:
        - Image 3.png
        - Image 4.png
    :param convert_label: convert folder name to a label
    :param dataset_path: Path to the dataset
    :return: List of (image file path, image class)
    """
    # Get all the folders in the full dataset path
    diseases = os.listdir(dataset_path)

    # Create a result dataset
    result = []

    # From each dataset, sample total_samples for train and test set
    for disease in diseases:
        # Get the full path
        disease_path = os.path.join(dataset_path, disease)

        # Check if not directory -> skip
        if not os.path.isdir(disease_path):
            continue

        # Get all the files in the folder
        all_images = os.listdir(disease_path)

        for image in all_images:
            clip_label = convert_label(disease)
            full_image_path = os.path.join(disease_path, image)
            result.append((full_image_path, clip_label))

    return result


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class ModelWithIntermediateLayers(torch.nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


class AugmentedDINOv2Dataset(Dataset):
    def __init__(
            self,
            image_paths: List[Union[str, Image.Image]],
            labels: List[str],
            num_augmentations: int = 1
    ):
        """
        Dataset for loading, preprocessing and augmenting images

        Args:
            image_paths: List of image paths or PIL Image objects
            labels: List of labels
            num_augmentations: Number of augmented copies to create per original image
        """
        self.image_paths = image_paths
        self.labels = labels
        self.num_augmentations = num_augmentations

        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            ResizeAndPad((256, 256), 14),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.based_transform = transforms.Compose([
            ResizeAndPad((256, 256), 14),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self) -> int:
        return len(self.image_paths) * (self.num_augmentations + 1)

    def __getitem__(self, idx: int):
        # Calculate which sample and which augmentation
        original_idx = idx // (self.num_augmentations + 1)
        aug_num = idx % (self.num_augmentations + 1)

        # Get original image path/object and label
        image = self.image_paths[original_idx]
        label = self.labels[original_idx]

        # Load image if it's a path
        if isinstance(image, str):
            temp = Image.open(image).convert('RGB')
            image = temp.copy()
            temp.close()

        # If it's the original image (aug_num = 0)
        if aug_num == 0:
            return self.based_transform(image), label

        # Apply augmentation for non-original images
        image = self.augment_transform(image)

        # Apply base transform if it exists
        return image, label


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        # Convert string labels to indices if needed
        unique_labels = sorted(list(set(train_loader.dataset.labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        label_indices = torch.tensor([label_to_idx[label] for label in labels]).to(device)

        # Move inputs to device
        if isinstance(inputs, (list, tuple)):
            inputs = [input.to(device) for input in inputs]
        else:
            inputs = inputs.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, label_indices)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        # Calculate top-1 and top-5 accuracy
        _, predicted = outputs.topk(5, 1, True, True)
        total += label_indices.size(0)

        # Top-1 accuracy
        correct_top1 += (predicted[:, 0] == label_indices).sum().item()

        # Top-5 accuracy
        correct_top5 += sum([1 for i, label in enumerate(label_indices)
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
            # Convert string labels to indices
            unique_labels = sorted(list(set(val_loader.dataset.labels)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            label_indices = torch.tensor([label_to_idx[label] for label in labels]).to(device)

            # Move inputs to device
            if isinstance(inputs, (list, tuple)):
                inputs = [input.to(device) for input in inputs]
            else:
                inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, label_indices)

            # Statistics
            running_loss += loss.item()

            # Calculate top-1 and top-5 accuracy
            _, predicted = outputs.topk(5, 1, True, True)
            total += label_indices.size(0)

            # Top-1 accuracy
            correct_top1 += (predicted[:, 0] == label_indices).sum().item()

            # Top-5 accuracy
            correct_top5 += sum([1 for i, label in enumerate(label_indices)
                                 if label in predicted[i]]).item()

    val_loss = running_loss / len(val_loader)
    val_acc_top1 = 100 * correct_top1 / total
    val_acc_top5 = 100 * correct_top5 / total

    return val_loss, val_acc_top1, val_acc_top5


class Dino(torch.nn.Module):
    def __init__(self, type, device):
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
            out_dim, use_n_blocks=1, use_avgpool=True, num_classes=100
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

    # Create the train and test dataloader
    train_dataset = AugmentedDINOv2Dataset(
        image_paths=train_paths,
        labels=train_labels,
        num_augmentations=19
    )

    test_dataset = AugmentedDINOv2Dataset(
        image_paths=test_paths,
        labels=test_labels,
        num_augmentations=1
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

    model = Dino("dinov2_vits14_reg", device)

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

