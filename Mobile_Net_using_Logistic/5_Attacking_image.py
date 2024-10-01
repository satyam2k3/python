# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 01:00:29 2024

@author: HP

STEP 5: attacking the original image to get the extracted watermark


ATTACKING THE IMAGE in different WAYS
"""

import torch
import numpy as np
import cv2
import random
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt

# Define attack functions
def fgsm_attack(image, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def add_salt_and_pepper_noise(image, amount=0.01, salt_vs_pepper=0.5):
    noisy_image = image.clone()
    num_channels, height, width = noisy_image.shape
    num_salt = int(np.ceil(amount * height * width * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * height * width * (1 - salt_vs_pepper)))
    
    for _ in range(num_salt):
        i = random.randint(0, height - 1)
        j = random.randint(0, width - 1)
        noisy_image[:, i, j] = 1
    
    for _ in range(num_pepper):
        i = random.randint(0, height - 1)
        j = random.randint(0, width - 1)
        noisy_image[:, i, j] = 0
    
    return noisy_image

def jpeg_compression_attack(image, quality=30):
    image_np = image.permute(1, 2, 0).numpy()
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    image_decoded = cv2.imdecode(encimg, 1)
    image_rgb = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    image_compressed = torch.tensor(image_rgb).permute(2, 0, 1).float() / 255.0
    return image_compressed

def geometric_transformation_attack(image):
    rotate = transforms.RandomRotation(degrees=90)
    transformed_image = rotate(image)
    return transformed_image


# Function to load image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()  # Convert the image to a tensor
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.squeeze(0)  # Remove batch dimension for simplicity


# Function to display images
def display_image(image_tensor):
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to HxWxC format
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

# Function to save image using PyTorch
def save_image_pytorch(image_tensor, output_path):
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension for saving
    utils.save_image(image_tensor, output_path, format="png")  # Save as PNG



# Main function that provides options to the user
def attack_image(image):
    print("Choose an attack to apply on the image:")
    print("1. FGSM Attack")
    print("2. Gaussian Noise")
    print("3. Salt-and-Pepper Noise")
    print("4. JPEG Compression")
    print("5. Geometric Transformation")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        epsilon = float(input("Enter the epsilon value for FGSM Attack (e.g., 0.1): "))
        gradient = torch.randn_like(image)  # Mock gradient for the example
        attacked_image = fgsm_attack(image, epsilon, gradient)
    elif choice == '2':
        std = float(input("Enter the standard deviation for Gaussian Noise (e.g., 0.1): "))
        attacked_image = add_gaussian_noise(image, std=std)
    elif choice == '3':
        amount = float(input("Enter the noise amount for Salt-and-Pepper Noise (e.g., 0.01): "))
        attacked_image = add_salt_and_pepper_noise(image, amount=amount)
    elif choice == '4':
        quality = int(input("Enter the JPEG quality for Compression Attack (e.g., 30): "))
        attacked_image = jpeg_compression_attack(image, quality=quality)
    elif choice == '5':
        attacked_image = geometric_transformation_attack(image)
    else:
        print("Invalid choice. No attack applied.")
        return image

    return attacked_image


# Load an image from disk
image_path = 'C:/Users/HP/OneDrive/Desktop/image_0067.png'
image = load_image(image_path)

# Display the original image
print("Original Image:")
display_image(image)

# Apply attack
attacked_image = attack_image(image)

# Display the attacked image
print("Attacked Image:")
display_image(attacked_image)

# Save the attacked image using PyTorch
save_path = "D:/Dataset_medical/Saved_Model/geometric.png"
save_image_pytorch(attacked_image, save_path)
print(f"Attacked image saved to {save_path}")
