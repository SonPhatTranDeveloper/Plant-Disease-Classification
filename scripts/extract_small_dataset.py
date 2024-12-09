"""
Author: Son Phat Tran
This file contains the code for extracting a smaller dataset from the full PlantVillage dataset
"""
import os
import shutil

import numpy as np


def extract_small_dataset(
        full_dataset_path: str,
        small_dataset_path: str,
        train_samples_per_class: int,
        test_samples_per_class: int) -> None:
    """
    From the full PlantVillage dataset, extract it into a smaller dataset with train and test set
    :param full_dataset_path: a path to the full PlantVillage dataset
    :param small_dataset_path: a path to the extracted small dataset
    :param train_samples_per_class: the number of samples for each disease (for train set)
    :param test_samples_per_class: the number of samples for each disease (for test set)
    :return: None
    """
    # Get all the folders in the full dataset path
    folders = os.listdir(full_dataset_path)

    # From each dataset, sample total_samples for train and test set
    for folder in folders:
        # Get the full path
        folder_path = os.path.join(full_dataset_path, folder)

        # Check if not directory -> skip
        if not os.path.isdir(folder_path):
            continue

        # Get all the files in the folder
        all_images = os.listdir(folder_path)

        # Shuffle the files
        np.random.shuffle(all_images)

        # Get the train and validation set
        train_set = all_images[:train_samples_per_class]
        test_set = all_images[train_samples_per_class:train_samples_per_class + test_samples_per_class]

        # Copy the files to the new destination
        train_set_dest = os.path.join(small_dataset_path, "train", folder)
        test_set_dest = os.path.join(small_dataset_path, "test", folder)

        # Make the directory
        if not os.path.exists(train_set_dest):
            os.makedirs(train_set_dest)

        if not os.path.exists(test_set_dest):
            os.makedirs(test_set_dest)

        # Copy the files over
        for file_name in train_set:
            file_path = os.path.join(folder_path, file_name)
            shutil.copy2(file_path, train_set_dest)

        for file_name in test_set:
            file_path = os.path.join(folder_path, file_name)
            shutil.copy2(file_path, test_set_dest)


if __name__ == "__main__":
    # Set the random seed for the process
    np.random.seed(1102)

    # Define the full folder path and target folder path
    full_path = "raw_dataset/full"
    small_path = "raw_dataset/small"
    extract_small_dataset(
        full_path, small_path,
        train_samples_per_class=10,
        test_samples_per_class=100
    )
