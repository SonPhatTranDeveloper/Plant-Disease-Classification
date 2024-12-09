"""
Author: Son Phat Tran
This file contains the logic for one-shot classification using ImageBind
"""
from typing import List

import torch
import torch.nn.functional as F
from imagebind import data
from imagebind.models import imagebind_model
from torchvision import transforms
from PIL import Image

from tqdm import tqdm

from utils.labelling import convert_label


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
            clip_label = convert_label(disease)
            full_image_path = os.path.join(disease_path, image)
            result.append((full_image_path, clip_label))

    return result


class ImageBindZeroShot:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

        # Standard image transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def encode_text(self, class_names: List[str]):
        """
        Encode class names into embeddings.

        Args:
            class_names: Name of the classes
        """
        text_inputs = data.load_and_transform_text(class_names, device=self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_inputs)
        return text_embeddings

    def encode_image(self, image_path: str):
        """
        Encode single image into embedding.

        Args:
            image_path: Path to the image
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        image_input = {
            "image": image_tensor
        }
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
        return image_embedding

    def classify_image(self, image_path: str, class_names: List[str], top_k: int = 5):
        """
        Classify image and return top-k predictions.

        Args:
            image_path: path to the image
            class_names: all the candidate class names
            top_k: How many candidate classes to suggest as label
        """
        # Get embeddings
        image_embedding = self.encode_image(image_path)
        text_embeddings = self.encode_text(class_names)

        # Calculate similarities
        similarities = F.cosine_similarity(
            image_embedding.unsqueeze(1),
            text_embeddings.unsqueeze(0),
            dim=2
        )

        # Get top-k predictions
        top_scores, top_indices = similarities[0].topk(min(top_k, len(class_names)))

        predictions = [
            (class_names[idx], score.item())
            for idx, score in zip(top_indices, top_scores)
        ]

        return predictions


def evaluate_accuracy(model, test_data, class_names):
    """
    Evaluate top-1 and top-5 accuracy on test data.

    Args:
        model: ImageBindZeroShot instance
        test_data: List of tuples (image_path, true_label)
        class_names: List of class names
    """
    top1_correct = 0
    top5_correct = 0
    total = len(test_data)

    for image_path, true_label in tqdm(test_data):
        predictions = model.classify_image(image_path, class_names, top_k=5)
        pred_classes = [p[0] for p in predictions]

        # Check top-1 accuracy
        if pred_classes[0] == true_label:
            top1_correct += 1

        # Check top-5 accuracy
        if true_label in pred_classes:
            top5_correct += 1

    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100

    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy
    }


if __name__ == "__main__":
    # Initialize model
    classifier = ImageBindZeroShot()

    # Example class names
    test_dataset = load_one_shot_dataset("raw_dataset/small/test")
    cand_labels = [MAPPING[label] for label in MAPPING]

    # Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy_results = evaluate_accuracy(classifier, test_dataset, cand_labels)
    print(f"\nTop-1 Accuracy: {accuracy_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {accuracy_results['top5_accuracy']:.2f}%")