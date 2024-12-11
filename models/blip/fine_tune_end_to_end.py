"""
Author: Son Phat Tran
This file contains the code for fine-tuning BLIP model
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from transformers import BlipProcessor, BlipModel

from PIL import Image
from tqdm import tqdm

from models.blip.one_shot import BLIPZeroShotClassifier, load_images
from utils.data import load_image_label_pairs
from utils.labelling import convert_label, MAPPING_WITH_PREFIX


def collate_fn(batch):
    # Get max sequence length in the batch
    max_length = max(x['input_ids'].size(0) for x in batch)

    # Initialize lists for batched elements
    pixel_values = []
    input_ids = []
    attention_mask = []

    for item in batch:
        pixel_values.append(item['pixel_values'])

        # Pad input_ids
        padded_ids = torch.zeros(max_length, dtype=item['input_ids'].dtype)
        padded_ids[:item['input_ids'].size(0)] = item['input_ids']
        input_ids.append(padded_ids)

        # Pad attention_mask
        padded_mask = torch.zeros(max_length, dtype=item['attention_mask'].dtype)
        padded_mask[:item['attention_mask'].size(0)] = item['attention_mask']
        attention_mask.append(padded_mask)

    # Stack all tensors
    return {
        'pixel_values': torch.stack(pixel_values),
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask)
    }


class ImageLabelDataset(Dataset):
    def __init__(self, image_paths, processor, labels, num_augmentations=10):
        self.num_augmentations = num_augmentations
        self.processor = processor
        self.image_paths = image_paths
        self.labels = labels

        # Define augmentation transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

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

        # Get the label
        label = self.labels[original_idx]

        # Process image and text separately
        image_inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=label,
            return_tensors="pt",
            padding=True
        )

        # Remove the batch dimension
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0)
        }


class BlipFineTuner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlipModel.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)

        # Unfreeze all parameters for end-to-end training
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.to(self.device)

    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=1e-5):
        # Create optimizer for all parameters
        optimizer = torch.optim.AdamW([
            {'params': self.model.vision_model.parameters(), 'lr': learning_rate},
            {'params': self.model.text_model.parameters(), 'lr': learning_rate},
            {'params': self.model.logit_scale, 'lr': learning_rate}
        ])

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass with loss computation
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )

                loss = outputs.loss

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Update progress bar with current loss
                pbar.set_postfix({
                    'loss': train_loss / len(train_loader)
                })

            # Validation
            val_loss = self.evaluate(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss
                }, 'weights/best_clip_model.pth')

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0

        pbar = tqdm(val_loader, desc='Evaluation')

        with torch.no_grad():
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_loss=True
                )

                loss = outputs.loss.item()
                total_loss += loss

                # Update progress bar with current loss
                pbar.set_postfix({
                    'loss': total_loss / len(val_loader)
                })

        # Perform zero-shot classification each time we do evaluation
        perform_one_shot_classification(self)

        return total_loss / len(val_loader)


# Example usage:
def main():
    # Initialize model
    fine_tuner = BlipFineTuner()

    # Load the train and test set
    train_pairs = load_image_label_pairs("raw_dataset/small/train", convert_label)
    train_image_paths = [item[0] for item in train_pairs]
    train_image_labels = [item[1] for item in train_pairs]

    test_pairs = load_image_label_pairs("raw_dataset/small/test", convert_label)
    test_image_paths = [item[0] for item in test_pairs]
    test_image_labels = [item[1] for item in test_pairs]

    # Create datasets
    train_dataset = ImageLabelDataset(
        image_paths=train_image_paths,
        labels=train_image_labels,
        processor=fine_tuner.processor,
        num_augmentations=19,
    )

    val_dataset = ImageLabelDataset(
        image_paths=test_image_paths,
        labels=test_image_labels,
        processor=fine_tuner.processor,
        num_augmentations=0
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Train the model
    fine_tuner.train(train_loader, val_loader, num_epochs=10, learning_rate=1e-7)


def perform_one_shot_classification(fine_tuner):
    # Initialize classifier
    classifier = BLIPZeroShotClassifier(
        model=fine_tuner.model,
        processor=fine_tuner.processor
    )

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


if __name__ == "__main__":
    # Set environment variable to avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
