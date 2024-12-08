"""
Author: Son Phat Tran
This file defines a simple model used for plant disease classification (using CNN)
"""
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


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model
    :param model: CNN model
    :param train_loader: data loader of the training set
    :param criterion: criterion (loss function)
    :param optimizer: optimizer used for training
    :param device: device used for training (cuda or cpu)
    :return: None
    """
    # Place the model in training mode
    model.train()

    # Iterate through batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Convert to device (copy to CUDA)
        data, target = data.to(device), target.to(device)

        # Clear out the previous gradient
        optimizer.zero_grad()

        # Calculate loss
        output = model(data)
        loss = criterion(output, target)

        # Backprop
        loss.backward()
        optimizer.step()

        # If batch size reach 100, 200, ... -> Perform evaluation
        if batch_idx % 100 == 0:
            print(f'Training Loss: {loss.item():.4f}')


def test_model(model, test_loader, device):
    """
    Test the model on test data
    :param model: CNN model
    :param test_loader: dataloader for the test data
    :param device: device for training (cuda or cpu)
    :return:
    """
    # Put the model into evaluation mode
    model.eval()

    # Calculate top-1 and top-5 accuracy
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # Calculate Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            top1_correct += (predicted == target).sum().item()

            # Calculate Top-5 accuracy
            _, top5_predicted = torch.topk(outputs.data, k=5, dim=1)
            for i, label in enumerate(target):
                if label in top5_predicted[i]:
                    top5_correct += 1

            total += target.size(0)

    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total

    print(f'Test Accuracy:')
    print(f'Top-1: {top1_accuracy:.2f}%')
    print(f'Top-5: {top5_accuracy:.2f}%')

    return top1_accuracy, top5_accuracy


def initialize_training(num_classes, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, device
