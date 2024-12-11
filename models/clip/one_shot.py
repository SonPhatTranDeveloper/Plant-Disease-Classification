"""
Author: Son Phat Tran
This file contains the code for CLIP zero-shot classification
"""
import os
from typing import List, Tuple, Dict

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from tqdm import tqdm

from utils.labelling import convert_label, MAPPING_WITH_PREFIX
from utils.data import load_image_label_pairs


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.

    Args:
        image_path (str): Local file path or URL of the image

    Returns:
        PIL.Image: Loaded image
    """
    # Avoid a bug in Pillow
    # by opening an image, make a copy then close it
    temp = Image.open(image_path)
    keep = temp.copy()
    temp.close()
    return keep


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
                 hypothesis_template: str = "{}") -> List[Tuple[str, float]]:
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
                       hypothesis_template: str = "{}",
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

        # Create progress bar for batch processing
        pbar = tqdm(total=len(images), desc="Processing images", unit="image")

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

            # Update progress bar
            pbar.update(len(batch_images))

        pbar.close()
        return all_results

    def calculate_accuracy(self,
                           images: List[Image.Image],
                           true_labels: List[str],
                           candidate_labels: List[str],
                           hypothesis_template: str = "{}",
                           batch_size: int = 32) -> Dict[str, float]:
        """
        Calculate top-1 and top-5 accuracy on a batch of images.

        Args:
            images (List[PIL.Image]): List of input images
            true_labels (List[str]): List of true labels for the images
            candidate_labels (List[str]): List of possible class labels
            hypothesis_template (str): Template string for text prompts
            batch_size (int): Size of batches for processing

        Returns:
            Dict[str, float]: Dictionary containing top-1 and top-5 accuracy scores
        """
        if len(images) != len(true_labels):
            raise ValueError("Number of images must match number of labels")

        if len(candidate_labels) < 5:
            raise ValueError("Need at least 5 candidate labels for top-5 accuracy")

        print("Calculating accuracy...")
        # Get predictions
        predictions = self.batch_classify(images, candidate_labels, hypothesis_template, batch_size)

        # Calculate top-1 accuracy
        top1_correct = sum(1 for pred, true_label in zip(predictions, true_labels)
                           if pred[0][0] == true_label)
        top1_accuracy = top1_correct / len(true_labels)

        # Calculate top-5 accuracy
        top5_correct = sum(1 for pred, true_label in zip(predictions, true_labels)
                           if true_label in [p[0] for p in pred[:5]])
        top5_accuracy = top5_correct / len(true_labels)

        # Calculate per-image accuracies for detailed analysis
        detailed_results = []
        for i, (pred, true_label) in enumerate(zip(predictions, true_labels)):
            top_5_predictions = [p[0] for p in pred[:5]]
            result = {
                'image_index': i,
                'true_label': true_label,
                'top_5_predictions': top_5_predictions,
                'top_5_probabilities': [p[1] for p in pred[:5]],
                'in_top_1': pred[0][0] == true_label,
                'in_top_5': true_label in top_5_predictions,
                'rank': next((i + 1 for i, p in enumerate(pred) if p[0] == true_label), -1)
            }
            detailed_results.append(result)

        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'detailed_results': detailed_results
        }


if __name__ == "__main__":
    # Initialize classifier
    classifier = CLIPZeroShotClassifier()

    # Load the dataset
    dataset = load_image_label_pairs("raw_dataset/small/test", convert_label)
    image_full_paths = [item[0] for item in dataset]
    true_labels = [item[1] for item in dataset]
    cand_labels = [MAPPING_WITH_PREFIX[label] for label in MAPPING_WITH_PREFIX]

    # Load the images
    loaded_images = load_images(image_full_paths)

    # Perform classification and calculate accuracy
    results = classifier.calculate_accuracy(loaded_images, true_labels, cand_labels)

    # Print results
    print("\nAccuracy Results:")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.3f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.3f}")
