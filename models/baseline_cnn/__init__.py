"""
Author: Son Phat Tran
This file defines a simple model used for plant disease classification (using CNN)
"""
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fifth convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the size after all conv layers
        # Input: 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Fourth block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Fifth block
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc='Training') as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss / total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(val_loader, desc='Validation') as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                running_loss += loss.item()

                # Top-1 accuracy
                _, predicted = torch.max(outputs.data, 1)
                top1_correct += (predicted == target).sum().item()

                # Top-5 accuracy
                _, top5_predicted = torch.topk(outputs.data, k=5, dim=1)
                for i, label in enumerate(target):
                    if label in top5_predicted[i]:
                        top5_correct += 1

                total += target.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{running_loss / total:.4f}',
                    'top1': f'{100. * top1_correct / total:.2f}%',
                    'top5': f'{100. * top5_correct / total:.2f}%'
                })

    val_loss = running_loss / len(val_loader)
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    return val_loss, top1_acc, top5_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=10, save_path='best_model.pth'):
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_top1_accs = []
    val_top5_accs = []

    print(f"Starting training on device: {device}")
    print(f"Number of epochs: {num_epochs}")

    for epoch in range(num_epochs):
        start_time = time.time()

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_top1_acc, val_top5_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_top1_accs.append(val_top1_acc)
        val_top5_accs.append(val_top5_acc)

        epoch_time = time.time() - start_time

        # Print epoch results
        print(f'\nEpoch Summary:')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Top-1 Acc: {val_top1_acc:.2f}%, Val Top-5 Acc: {val_top5_acc:.2f}%')

        # Save best model
        if val_top1_acc > best_val_acc:
            best_val_acc = val_top1_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f'Saved new best model with validation accuracy: {best_val_acc:.2f}%')

    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_top1_accs': val_top1_accs,
        'val_top5_accs': val_top5_accs,
        'best_val_acc': best_val_acc
    }


def initialize_training(num_classes, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, device
