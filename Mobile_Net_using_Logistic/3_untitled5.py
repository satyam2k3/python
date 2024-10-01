# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:20:20 2024

@author: HP
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Logistic map function to generate chaotic sequence
def logistic_map(x0, r, steps):
    chaotic_sequence = []
    x = x0
    for _ in range(steps):
        x = r * x * (1 - x)  # Logistic map equation
        chaotic_sequence.append(x)
    
    chaotic_sequence = np.array(chaotic_sequence)
    return chaotic_sequence

# Function to binarize the chaotic sequence
def binarize_chaotic_sequence(sequence, height, width):
    sequence_flat = sequence.flatten()
    min_val = np.min(sequence_flat)
    max_val = np.max(sequence_flat)
    normalized_sequence = (sequence_flat - min_val) / (max_val - min_val)  # Normalize to 0-1 range
    
    binary_sequence = np.where(normalized_sequence > 0.5, 1, 0)  # Thresholding to get binary values
    binary_matrix = binary_sequence[:height * width].reshape(height, width)  # Reshape to image size
    return binary_matrix

# XOR-based encryption using the chaos matrix
def apply_chaos_matrix(image, chaos_matrix):
    encrypted_image = np.bitwise_xor(image, chaos_matrix)
    return encrypted_image

# Arnold Cat Map function for scrambling the image
def arnold_cat_map(image, a, b, iterations):
    height, width = image.shape
    new_image = np.copy(image)
    
    for _ in range(iterations):
        transformed_image = np.zeros_like(image)
        for x in range(height):
            for y in range(width):
                new_x = (x + a * y) % height
                new_y = (b * x + y) % width
                transformed_image[new_x, new_y] = new_image[x, y]
        new_image = transformed_image
    
    return new_image

# Binarization of image
def binarize_image(image, threshold=128):
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image

# Load the watermark image
image_path = "C:/Users/HP/OneDrive/Desktop/images.png"
watermark = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize and get image dimensions
height, width = watermark.shape

# Step 1: Generate logistic chaotic sequence
x0 = 0.5  # Initial condition for logistic map
r = 3.9   # Control parameter for logistic map (values in [3.57, 4] are chaotic)
steps = height * width
chaotic_sequence = logistic_map(x0, r, steps)

# Step 2: Binarize the chaotic sequence to create the chaos matrix
chaos_matrix = binarize_chaotic_sequence(chaotic_sequence, height, width)

# Step 3: Apply XOR encryption using the chaos matrix
encrypted_watermark = apply_chaos_matrix(watermark, chaos_matrix)

# Step 4: Apply Arnold Cat Map for further scrambling
a_arnold = 3
b_arnold = 5
iterations = 10
scrambled_watermark = arnold_cat_map(encrypted_watermark, a_arnold, b_arnold, iterations)

# Binarize the scrambled watermark
binarized_scrambled_watermark = binarize_image(scrambled_watermark)
print(binarized_scrambled_watermark.shape)

# Convert to PyTorch tensor
binarized_scrambled_watermark_tensor = torch.tensor(binarized_scrambled_watermark, dtype=torch.uint8)
torch.save(binarized_scrambled_watermark_tensor, 'D:/__Mobnet/binarized_scrambled_watermark.pt')

# Display the encrypted and scrambled watermark
plt.figure(figsize=(8, 8))
plt.imshow(binarized_scrambled_watermark, cmap='gray')
plt.title('Encrypted Watermark using Logistic Map and Arnold Map')
plt.axis('off')
plt.show()
