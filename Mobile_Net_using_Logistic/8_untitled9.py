# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 00:23:29 2024

@author: HP
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Logistic map function (already defined in encryption)
def logistic_map(x0, r, steps):
    chaotic_sequence = []
    x = x0
    for _ in range(steps):
        x = r * x * (1 - x)
        chaotic_sequence.append(x)
    
    chaotic_sequence = np.array(chaotic_sequence)
    return chaotic_sequence

# Function to binarize the chaotic sequence (already defined in encryption)
def binarize_chaotic_sequence(sequence, height, width):
    sequence_flat = sequence.flatten()
    min_val = np.min(sequence_flat)
    max_val = np.max(sequence_flat)
    normalized_sequence = (sequence_flat - min_val) / (max_val - min_val)
    
    binary_sequence = np.where(normalized_sequence > 0.5, 1, 0)
    binary_matrix = binary_sequence[:height * width].reshape(height, width)
    return binary_matrix

# XOR-based decryption using the same chaos matrix
def apply_chaos_matrix(image, chaos_matrix):
    decrypted_image = np.bitwise_xor(image, chaos_matrix)
    return decrypted_image

# Inverse Arnold Cat Map for unscrambling the image
def inverse_arnold_cat_map(image, a, b, iterations):
    height, width = image.shape
    new_image = np.copy(image)
    
    for _ in range(iterations):
        transformed_image = np.zeros_like(image)
        for x in range(height):
            for y in range(width):
                new_x = ((x - a * y) % height + height) % height
                new_y = ((y - b * x) % width + width) % width
                transformed_image[x, y] = new_image[new_x, new_y]
        new_image = transformed_image
    
    return new_image

# Binarization of image (already defined in encryption)
def binarize_image(image, threshold=128):
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image

# Load the encrypted watermark (to decrypt)
encrypted_image_path = "D:/__Mobnet/extracted_attack_watermark.pt"
encrypted_watermark_tensor = torch.load(encrypted_image_path)
encrypted_watermark = encrypted_watermark_tensor.numpy()

# Image dimensions
height, width = encrypted_watermark.shape

# Generate the same logistic chaotic sequence for decryption
x0 = 0.5  # Initial condition for logistic map (same as encryption)
r = 3.9   # Control parameter for logistic map (same as encryption)
steps = height * width
chaotic_sequence = logistic_map(x0, r, steps)

# Binarize the chaotic sequence to generate the same chaos matrix
chaos_matrix = binarize_chaotic_sequence(chaotic_sequence, height, width)

# Reverse Arnold Cat Map (decrypt)
a_arnold = 3
b_arnold = 5
iterations = 10
unscrambled_watermark = inverse_arnold_cat_map(encrypted_watermark, a_arnold, b_arnold, iterations)

# XOR decryption (reverse encryption)
decrypted_watermark = apply_chaos_matrix(unscrambled_watermark, chaos_matrix)

# Display the decrypted watermark
plt.figure(figsize=(8, 8))
plt.imshow(decrypted_watermark, cmap='gray')
plt.title('Decrypted Watermark')
plt.axis('off')  # Hide axes
plt.show()

# Save the decrypted watermark if needed
cv2.imwrite('D:/__Mobnet/decrypted_watermark.png', decrypted_watermark * 255)
