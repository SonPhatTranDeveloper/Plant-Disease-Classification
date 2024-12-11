"""
Author: Son Phat Tran
This file contains the utility code for data handling
"""
import os

from typing import Tuple, List, Any


def load_image_label_pairs(dataset_path: str, convert_label: Any) -> List[Tuple[str, str]]:
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
