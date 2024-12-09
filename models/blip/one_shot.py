"""
Author: Son Phat Tran
This code contains the logic for one-shot classification using BLIP
"""
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import BlipProcessor, BlipForImageTextRetrieval
from torchvision import datasets, transforms

from tqdm import tqdm
from pathlib import Path
from PIL import Image

from utils.labelling import MAPPING, convert_label


class BLIPZeroShotClassifier:
    def __init__(self, model_name="Salesforce/blip-itm-base-coco"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)

    def prepare_text_inputs(self, categories: List[str]):
        """
        Prepare text inputs for all categories.

        Args:
            categories: List of candidate labels
        """
        text_inputs = []
        for category in categories:
            encoded = self.processor(text=convert_label(category), return_tensors="pt", padding=True)
            text_inputs.append(encoded)
        return text_inputs

    def classify_image(self, image_path: str, categories: List[str], return_top_k=1):
        """
        Perform zero-shot classification on a single image.

        Args:
            image_path: Path to the image
            categories: candidate labels
            return_top_k: how many candidate labels to return
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Prepare text inputs for all categories
        scores = []
        text_inputs = self.prepare_text_inputs(categories)

        # Get similarity scores for each category
        with torch.no_grad():
            for text_input in text_inputs:
                text_input = {k: v.to(self.device) for k, v in text_input.items()}
                outputs = self.model(**image_inputs, **text_input)
                itm_score = F.softmax(outputs.itm_score, dim=1)[:, 1]
                scores.append(itm_score.item())

        # Get top-k predictions
        scores = torch.tensor(scores)
        top_k_scores, top_k_indices = torch.topk(scores, min(return_top_k, len(categories)))

        predictions = [(categories[idx], scores[idx].item()) for idx in top_k_indices]
        return predictions


def evaluate_zero_shot(blip_classifier, dataset_path, categories, batch_size=32):
    """
    Evaluate the zero-shot classifier on a dataset.

    Args:
        blip_classifier: BLIP classifier
        dataset_path: directory of the dataset
        categories: candidate labels
        batch_size: how many images to process at a time
    """
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        for img, label in zip(images, labels):
            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(img)

            # Save temporary image
            temp_path = "temp.jpg"
            img_pil.save(temp_path)

            # Get predictions
            predictions = blip_classifier.classify_image(temp_path, categories, return_top_k=5)

            # Get ground truth category
            true_category = categories[label]

            # Check top-1 accuracy
            if predictions[0][0] == true_category:
                top1_correct += 1

            # Check top-5 accuracy
            pred_categories = [p[0] for p in predictions]
            if true_category in pred_categories:
                top5_correct += 1

            total += 1

    # Clean up temporary file
    Path("temp.jpg").unlink(missing_ok=True)

    # Calculate accuracies
    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100

    return {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "total_samples": total
    }


# Test code
if __name__ == "__main__":
    # Initialize classifier
    classifier = BLIPZeroShotClassifier()

    # Example categories (modify based on your dataset)
    candidate_labels = [label for label in MAPPING]

    # Evaluate on test dataset
    data_dir = "raw_dataset/small/test"
    results = evaluate_zero_shot(classifier, data_dir, candidate_labels)

    print("\nEvaluation Results:")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"Total samples evaluated: {results['total_samples']}")
