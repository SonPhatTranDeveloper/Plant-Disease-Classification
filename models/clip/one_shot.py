"""
Author: Son Phat Tran
This file contains the code for CLIP-2 zero-shot classification
"""
import os
from typing import List, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# Define the mapping between labels and one-shot prompt
MAPPING = {
    "Apple__apple_scab": "An image of an apple leaf infected with apple scab disease",
    "Apple__black_rot": "An image of an apple leaf affected by black rot disease",
    "Apple__cedar_apple_rust": "An image of an apple leaf showing cedar apple rust infection",
    "Apple__healthy": "An image of a healthy apple tree leaf",
    "Background": "An image of a background or non-plant material",
    "Blueberry__healthy": "An image of a healthy blueberry plant leaf",
    "Cherry__healthy": "An image of a healthy cherry tree leaf",
    "Cherry__powdery_mildew": "An image of a cherry leaf infected with powdery mildew",
    "Corn__cercospora_leaf_spot_gray_leaf_spot": "An image of a corn leaf with gray leaf spot disease",
    "Corn__common_rust": "An image of a corn leaf infected with common rust",
    "Corn__healthy": "An image of a healthy corn plant leaf",
    "Corn__northern_leaf_blight": "An image of a corn leaf affected by northern leaf blight",
    "Grape__black_rot": "An image of a grape leaf with black rot disease",
    "Grape__esca_(black_measles)": "An image of a grape leaf showing esca disease symptoms",
    "Grape__healthy": "An image of a healthy grape vine leaf",
    "Grape__leaf_blight_(isariopsis_leaf_spot)": "An image of a grape leaf with isariopsis leaf spot disease",
    "Orange__haunglongbing_(citrus_greening)": "An image of an orange tree leaf affected by citrus greening disease",
    "Peach__bacterial_spot": "An image of a peach leaf with bacterial spot infection",
    "Peach__healthy": "An image of a healthy peach tree leaf",
    "Pepper_bell__bacterial_spot": "An image of a bell pepper leaf with bacterial spot disease",
    "Pepper_bell__healthy": "An image of a healthy bell pepper plant leaf",
    "Potato__early_blight": "An image of a potato leaf affected by early blight disease",
    "Potato__healthy": "An image of a healthy potato plant leaf",
    "Potato__late_blight": "An image of a potato leaf infected with late blight disease",
    "Raspberry__healthy": "An image of a healthy raspberry plant leaf",
    "Soybean__healthy": "An image of a healthy soybean plant leaf",
    "Squash__powdery_mildew": "An image of a squash leaf infected with powdery mildew",
    "Strawberry__healthy": "An image of a healthy strawberry plant leaf",
    "Strawberry__leaf_scorch": "An image of a strawberry leaf with leaf scorch disease",
    "Tomato__bacterial_spot": "An image of a tomato leaf with bacterial spot infection",
    "Tomato__early_blight": "An image of a tomato leaf affected by early blight disease",
    "Tomato__healthy": "An image of a healthy tomato plant leaf",
    "Tomato__late_blight": "An image of a tomato leaf infected with late blight disease",
    "Tomato__leaf_mold": "An image of a tomato leaf showing leaf mold infection",
    "Tomato__mosaic_virus": "An image of a tomato leaf infected with mosaic virus",
    "Tomato__septoria_leaf_spot": "An image of a tomato leaf with septoria leaf spot disease",
    "Tomato__spider_mites_two_spotted_spider_mite": "An image of a tomato leaf damaged by two-spotted spider mites",
    "Tomato__target_spot": "An image of a tomato leaf with target spot disease",
    "Tomato__yellow_leaf_curl_virus": "An image of a tomato leaf infected with yellow leaf curl virus"
}


def create_clip_label(label: str) -> str:
    """
    Get the label for CLIP one-shot classification using image label
    :param label: class label
    :return: CLIPv2 one-shot label
    """
    return MAPPING[label]


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
            clip_label = create_clip_label(disease)
            full_image_path = os.path.join(disease_path, image)
            result.append((full_image_path, clip_label))

    return result


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.

    Args:
        image_path (str): Local file path or URL of the image

    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(image_path)


def load_images(image_paths: List[str]) -> List[Image.Image]:
    """
        Load multiple images from a file path or URL.

        Args:
            image_paths (List[str]): Local file paths or URL of the image

        Returns:
            PIL.Image: Loaded image
        """
    return [load_image(path) for path in image_paths]


class CLIPZeroShotClassifier:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize the CLIP zero-shot classifier.

        Args:
            model_name (str): Name of the CLIP model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify(self,
                 image: Image.Image,
                 candidate_labels: List[str],
                 hypothesis_template: str = "a photo of a {}") -> List[Tuple[str, float]]:
        """
        Perform zero-shot classification on an image.

        Args:
            image (PIL.Image): Input image to classify
            candidate_labels (List[str]): List of possible class labels
            hypothesis_template (str): Template string for text prompts

        Returns:
            List[Tuple[str, float]]: List of (label, probability) pairs, sorted by probability
        """
        return self.batch_classify([image], candidate_labels, hypothesis_template)[0]

    def batch_classify(self,
                       images: List[Image.Image],
                       candidate_labels: List[str],
                       hypothesis_template: str = "a photo of a {}",
                       batch_size: int = 32) -> List[List[Tuple[str, float]]]:
        """
        Perform zero-shot classification on a batch of images.

        Args:
            images (List[PIL.Image]): List of input images to classify
            candidate_labels (List[str]): List of possible class labels
            hypothesis_template (str): Template string for text prompts
            batch_size (int): Size of batches for processing

        Returns:
            List[List[Tuple[str, float]]]: List of classification results for each image
        """
        text_prompts = [hypothesis_template.format(label) for label in candidate_labels]
        all_results = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Prepare inputs
            inputs = self.processor(
                text=text_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Calculate features
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()

            # Process results for each image in batch
            for image_probs in probs:
                results = [(label, float(prob)) for label, prob in zip(candidate_labels, image_probs)]
                all_results.append(sorted(results, key=lambda x: x[1], reverse=True))

        return all_results

    def calculate_accuracy(self,
                           images: List[Image.Image],
                           true_labels: List[str],
                           candidate_labels: List[str],
                           hypothesis_template: str = "a photo of a {}",
                           batch_size: int = 32) -> float:
        """
        Calculate classification accuracy on a batch of images.

        Args:
            images (List[PIL.Image]): List of input images
            true_labels (List[str]): List of true labels for the images
            candidate_labels (List[str]): List of possible class labels
            hypothesis_template (str): Template string for text prompts
            batch_size (int): Size of batches for processing

        Returns:
            float: Classification accuracy (0-1)
        """
        if len(images) != len(true_labels):
            raise ValueError("Number of images must match number of labels")

        # Get predictions
        predictions = self.batch_classify(images, candidate_labels, hypothesis_template, batch_size)
        predicted_labels = [result[0][0] for result in predictions]  # Get top prediction for each image

        # Calculate accuracy
        correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
        accuracy = correct_predictions / len(true_labels)

        return accuracy


if __name__ == "__main__":
    # Initialize classifier
    classifier = CLIPZeroShotClassifier()

    # Load the dataset
    dataset = load_one_shot_dataset("raw_dataset/small/test")
    image_full_paths = [item[0] for item in dataset]
    true_labels = [item[1] for item in dataset]
    candidate_labels = [MAPPING[label] for label in MAPPING]

    # Load the images
    images = load_images(image_full_paths)

    # Perform classification and calculate accuracy
    accuracy = classifier.calculate_accuracy(images, true_labels, candidate_labels)
    print(f"\nClassification Accuracy: {accuracy:.3f}")
