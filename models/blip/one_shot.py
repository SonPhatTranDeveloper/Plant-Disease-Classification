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
        self.model.eval()  # Set model to evaluation mode

    def prepare_text_inputs(self, categories):
        """Prepare text inputs for all categories at once."""
        texts = [f"a photo of a {category}" for category in categories]
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in text_inputs.items()}

    @torch.no_grad()
    def classify_batch(self, images, categories, return_top_k=1):
        """Perform zero-shot classification on a batch of images."""
        batch_size = images.size(0)
        num_categories = len(categories)

        # Process images
        images = [transforms.ToPILImage()(img) for img in images]
        image_inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Process text once for all categories
        text_inputs = self.prepare_text_inputs(categories)

        # Initialize scores tensor
        all_scores = torch.zeros(batch_size, num_categories).to(self.device)

        # Calculate similarity scores for each category
        for i, category in enumerate(categories):
            # Select the text inputs for current category
            curr_text_inputs = {
                k: v[i:i + 1] for k, v in text_inputs.items()
            }

            # Get model outputs
            outputs = self.model(**image_inputs, **curr_text_inputs)
            scores = F.softmax(outputs.itm_score, dim=1)[:, 1]
            all_scores[:, i] = scores

        # Get top-k predictions
        top_k_scores, top_k_indices = torch.topk(all_scores, k=min(return_top_k, num_categories), dim=1)

        return top_k_scores, top_k_indices


def evaluate_zero_shot(classifier, data_dir, categories, batch_size=32):
    """Evaluate the zero-shot classifier on a dataset."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(classifier.device)
        labels = labels.to(classifier.device)

        # Get predictions for the entire batch
        top_k_scores, top_k_indices = classifier.classify_batch(images, categories, return_top_k=5)

        # Calculate top-1 accuracy
        top1_preds = top_k_indices[:, 0]
        top1_correct += (top1_preds == labels).sum().item()

        # Calculate top-5 accuracy
        top5_correct += sum(labels.view(-1, 1) == top_k_indices).any(dim=1).sum().item()

        total += labels.size(0)

    return {
        "top1_accuracy": (top1_correct / total) * 100,
        "top5_accuracy": (top5_correct / total) * 100,
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
