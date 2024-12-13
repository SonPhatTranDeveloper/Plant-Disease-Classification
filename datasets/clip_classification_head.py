"""
Author: Son Phat Tran
This file contains the code for classification label
"""

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


NO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


CROP_AND_ROTATION_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageLabelDataset(Dataset):
    def __init__(self, image_paths, labels, num_augmentations=10, transform=CROP_AND_ROTATION_TRANSFORM):
        self.num_augmentations = num_augmentations
        self.image_paths = image_paths
        self.labels = labels

        # Define augmentation transforms
        self.augment_transform = transform

        # Mapping between labels
        sorted_labels = list(sorted(list(set(labels))))
        self.label_mapping = {
            label: index for index, label in enumerate(sorted_labels)
        }

    def __len__(self):
        return len(self.image_paths) * (self.num_augmentations + 1)

    def __getitem__(self, idx):
        # Get the augmentation index
        original_idx = idx // (self.num_augmentations + 1)
        aug_num = idx % (self.num_augmentations + 1)

        # Load the image
        image_path = self.image_paths[original_idx]
        temp = Image.open(image_path).convert('RGB')
        image = temp.copy()
        temp.close()

        # Check if original image
        if aug_num != 0:
            image = self.augment_transform(image)
        else:
            image = NO_TRANSFORM(image)

        # Get the label
        label = self.labels[original_idx]
        label = self.label_mapping[label]

        return image, label