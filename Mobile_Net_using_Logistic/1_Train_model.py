# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:44:12 2024

@author: HP

STEP 1 : Training the MobileNetV3 model on a custom dataset 
TRAINING THE MODEL AND SAVING IT 
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import random

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Parameters
image_dir = "D:/Dataset_medical/new_dataset"
val_dir = "D:/Dataset_medical/Medical_dataset/val"
test_dir = "D:/Dataset_medical/Medical_dataset/test"
learning_rate = 0.001
batch_size = 30
num_epochs = 4

model_save_path = "D:/Dataset_medical/Saved_Model/mobilenet_v3_large.pth"

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
val_dataset = CustomImageDataset(image_dir=val_dir, transform=transform)
test_dataset = CustomImageDataset(image_dir=test_dir, transform=transform)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load MobileNetV3 Model
model = models.mobilenet_v3_large(weights=True)


# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training and Validation Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    # Training Phase
    model.train()  # Set the model to training mode
    train_loss = 0.0
    for images in data_loader:
        optimizer.zero_grad()

        outputs = model(images)
        labels = torch.randint(0, 2, (images.size(0),))  # Replace with real labels
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Training Loss: {train_loss / len(data_loader)}")

    # Validation Phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for images in val_loader:
            outputs = model(images)
            labels = torch.randint(0, 2, (images.size(0),))  # Replace with real labels
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")

print("Training complete.")


# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Testing Phase
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        labels = torch.randint(0, 2, (images.size(0),))  # Replace with real labels
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Loss: {test_loss / len(test_loader)}")
print(f"Test Accuracy: {100 * correct / total}%")