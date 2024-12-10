"""
Author: Son Phat Tran
This file contains the logic of creating the dataset for DINOv2
"""
import os
from typing import List, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from PIL import Image

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
        img = T.Resize(self.target_size, interpolation=InterpolationMode.BICUBIC)(img)

        # Calculate padding
        pad_width = (self.multiple - img.width % self.multiple) % self.multiple
        pad_height = (self.multiple - img.height % self.multiple) % self.multiple

        # Apply padding
        img = T.Pad(
            (pad_width // 2,
             pad_height // 2,
             pad_width - pad_width // 2,
             pad_height - pad_height // 2)
        )(img)

        return img


STANDARD_TRANSFORM = T.Compose([
    ResizeAndPad((256, 256), 14),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])


def load_one_shot_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """
    Load the oneshot dataset classification (image + label) from a folder
    Note that the folder must have the following structure:
    - class 1:
        - Image 1.png
        - Image 2.png
    - class 2:
        - Image 3.png
        - Image 4.png
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
            label = disease
            full_image_path = os.path.join(disease_path, image)
            result.append((full_image_path, label))

    return result


class DINOv2Dataset(Dataset):
    def __init__(self, image_paths: List[Union[str, Image.Image]], transform=STANDARD_TRANSFORM):
        """
        Dataset for loading and preprocessing images

        Args:
            image_paths: List of image paths or PIL Image objects
            transform: Torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.image_paths[idx]

        if isinstance(image, str):
            temp = Image.open(image).convert('RGB')
            image = temp.copy()
            temp.close()

        if self.transform:
            image = self.transform(image)

        return image


def create_dataloader(dataset_path: str, batch_size: int = 32, num_workers: int = 4):
    """
    Create DINOv2 Image dataset
    :param dataset_path: path to the images
    :param batch_size: batch size of the dataloader
    :param num_workers: number of workers for dataloader
    :return:
    """
    # Load image path and labels
    paths_and_labels = load_one_shot_dataset(dataset_path)

    # Extract the paths and labels
    paths = [item[0] for item in paths_and_labels]
    labels = [item[1] for item in paths_and_labels]

    # Create dataset and dataloader
    dataset = DINOv2Dataset(paths)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return dataloader, labels
