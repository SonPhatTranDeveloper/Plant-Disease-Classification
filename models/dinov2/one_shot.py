"""
Author: Son Phat Tran
This file contains the logic of performing one-shot classification using DINOv2 embedding
"""
from typing import List, Dict, Any
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from models.dinov2 import vit_small
from datasets.dinov2 import create_dataloader


class DINOv2Embeddings:
    def __init__(self):
        # Create DINOv2 model
        model = vit_small(patch_size=14,
                          img_size=526,
                          init_values=1.0,
                          num_register_tokens=4,
                          block_chunks=0)

        self.embedding_size = 384
        self.number_of_heads = 6

        # Load the weights
        model.load_state_dict(
            torch.load("models/dinov2/weights/dinov2_vits14_reg4_pretrain.pth")
        )

        # Create a copy of DINOv2 model
        self.model = deepcopy(model)

        # Create device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def generate_embeddings_batch(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Generate embeddings for all images in folder

        Args:
            data_loader: DataLoader that contain images
        Returns:
            Tensor containing embeddings for all images
        """
        total_images = len(data_loader.dataset)
        embeddings = []

        with tqdm(total=total_images, desc="Generating embeddings", unit='img') as pbar:
            for batch in data_loader:
                batch = batch.to(self.device, non_blocking=True)
                batch_embeddings = self.model(batch)
                embeddings.append(batch_embeddings.cpu())

                pbar.update(batch.size(0))  # Update by actual batch size

                # Optional: add memory usage to progress bar
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_reserved() / 1E9
                    pbar.set_postfix({'GPU Memory': f'{mem:.2f}GB'})

        return torch.cat(embeddings, dim=0)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalize embeddings to unit length

    Args:
        embeddings: Tensor of shape (N, embedding_dim)
    Returns:
        Normalized embeddings
    """
    return F.normalize(embeddings, p=2, dim=1)


class DINOv2OneShot:
    def __init__(self):
        """
        Initialize the one-shot classifier using DINOv2 embeddings
        """
        self.nn_model = None
        self.support_embeddings = None
        self.support_labels = None
        self.label_to_classname = None

    def fit(self, support_embeddings: torch.Tensor, support_labels: List[str],
            label_to_classname: Dict[str, str] = None):
        """
        Fit the one-shot classifier with support set

        Args:
            support_embeddings: Tensor of shape (N, embedding_dim) containing DINOv2 embeddings
            support_labels: List of labels corresponding to the embeddings
            label_to_classname: Optional dictionary mapping label IDs to class names
        """
        self.support_embeddings = normalize_embeddings(support_embeddings)
        self.support_labels = support_labels
        self.label_to_classname = label_to_classname

        # Initialize nearest neighbor model
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn_model.fit(self.support_embeddings.cpu().numpy())

    def predict(self, query_embeddings: torch.Tensor, k: int = 1) -> tuple[list[list[Any]], Any]:
        """
        Predict labels for query embeddings

        Args:
            query_embeddings: Tensor of shape (N, embedding_dim)
            k: Number of nearest neighbors to consider (for top-k predictions)
        Returns:
            Tuple of (predicted labels, distances)
        """
        query_embeddings = normalize_embeddings(query_embeddings)

        # Find k nearest neighbors
        distances, indices = self.nn_model.kneighbors(
            query_embeddings.cpu().numpy(),
            n_neighbors=k
        )

        # Get predicted labels
        predictions = []
        for idx_list in indices:
            neighbor_labels = [self.support_labels[idx] for idx in idx_list]
            predictions.append(neighbor_labels)

        return predictions, distances

    def evaluate(self, query_embeddings: torch.Tensor, query_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate the model using top-1 and top-5 accuracy

        Args:
            query_embeddings: Tensor of shape (N, embedding_dim)
            query_labels: List of true labels
        Returns:
            Dictionary containing top-1 and top-5 accuracy
        """
        predictions, _ = self.predict(query_embeddings, k=5)

        # Calculate top-1 accuracy
        top1_correct = sum(1 for pred, true in zip(predictions, query_labels)
                           if pred[0] == true)
        top1_accuracy = top1_correct / len(query_labels)

        # Calculate top-5 accuracy
        top5_correct = sum(1 for pred, true in zip(predictions, query_labels)
                           if true in pred)
        top5_accuracy = top5_correct / len(query_labels)

        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy
        }


if __name__ == "__main__":
    # Create dataloader and labels for train and test data
    train_dataloader, train_labels = create_dataloader("raw_dataset/small/train")
    test_dataloader, test_labels = create_dataloader("raw_dataset/small/test")

    # Create embedding generator
    dino_v2_embedding = DINOv2Embeddings()

    # Generate the embedding of the train set
    print("Generating embeddings for support (train) set ...")
    train_embeddings = dino_v2_embedding.generate_embeddings_batch(train_dataloader)
    print("Generating embeddings for test set ...")
    test_embeddings = dino_v2_embedding.generate_embeddings_batch(test_dataloader)

    # Create classifier
    classifier = DINOv2OneShot()
    classifier.fit(train_embeddings, train_labels)

    # Evaluate
    metrics = classifier.evaluate(test_embeddings, test_labels)
    print("Generating one-shot classification result")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.3f}")
