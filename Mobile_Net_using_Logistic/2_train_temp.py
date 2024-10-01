# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 02:25:06 2024

@author: HP

STEP 2 : this file is same as temp.py but this is using the trained model ( updating this file by unfreezing last two layers)

FOR LOADING AND CHECKING OUTPUTS ( temp.py but when model is being trained)
"""

import torch 
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.fftpack import dct
import numpy as np

# Load the MobileNetV3-Large model
model = models.mobilenet_v3_large(weights=False)  # Set weights=False because we'll load our own weights

# Load the saved model weights
model_save_path = "D:/Dataset_medical/Saved_Model/mobilenet_v3_large.pth"
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

# Modify the classifier
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
)

# Freeze all layers except the new classifier
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers of the classifier
classifier = model.classifier
num_layers = len(classifier)

# If the classifier has more than two layers (typically, it has just one Linear layer here):
if num_layers > 1:
    # Unfreeze the last two layers
    for param in classifier[-2].parameters():
        param.requires_grad = True
    for param in classifier[-1].parameters():
        param.requires_grad = True

# Example input tensor (batch of images)
image_path = "D:/Dataset_medical/Medical_dataset/test/image_0028.jpg"
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a tensor (H, W, C) -> (C, H, W)
])

input_tensor = transform(image)

# Add a batch dimension
input_tensor = input_tensor.unsqueeze(0)  # Shape becomes (1, 3, 224, 224)

# Extract 128-Dimensional Features
features = model(input_tensor)
print("Original Features Shape:", features.shape)  # Output: torch.Size([1, 128])

# Convert PyTorch Tensor to NumPy Array and then apply DCT 
features_np = features.detach().numpy()
dct_features_np = dct(features_np, type=2, norm='ortho')
print(dct_features_np)

dct_features_flat = dct_features_np.flatten()
print(dct_features_flat)

# top_32_indices = np.argsort(np.abs(dct_features_flat))[-32:]
# print(top_32_indices)

# # Select the 32 coefficients with the largest magnitudes
# top_32_coeffs = dct_features_flat[top_32_indices]
# print(top_32_coeffs)
 
top_32_coeffs = dct_features_flat[:32]  # First 32 low-frequency coefficients
print("Top 32 Low-Frequency Coefficients:", top_32_coeffs)

sign_transformed_coeffs = np.where(top_32_coeffs > 0, 1, 0)
print(sign_transformed_coeffs)
print(type(sign_transformed_coeffs))  # numpy array

# Convert the Transformed Coefficients Back to PyTorch Tensor
sign_transformed_coeffs_tensor = torch.tensor(sign_transformed_coeffs)

# Save the DCT-transformed features to a file
save_path = "D:/Dataset_medical/Saved_Model/hashing_values.pt"
torch.save(dct_features_np, save_path)
print(f"DCT features saved to {save_path}")

# # Optionally, you can also save the original 128-dimensional features directly
# original_features_save_path = "D:/__Mobnet/original_features.pt"
# torch.save(features, original_features_save_path)
# print(f"Original features saved to {original_features_save_path}")
