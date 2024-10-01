# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 01:57:46 2024

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:20:20 2024

@author: HP
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Logistic map function to encrypt the image
def logistic_map_encrypt(image, k2, iterations):
    """Encrypt the image using Logistic Map."""
    height, width = image.shape
    image = image / 255.0  # Normalize to [0, 1]
    r = k2  # Logistic map parameter
    
    def logistic_map(x, r):
        return r * x * (1 - x)
    
    encrypted_image = np.copy(image)
    for _ in range(iterations):
        for i in range(height):
            for j in range(width):
                encrypted_image[i, j] = logistic_map(image[i, j], r)
    
    return (encrypted_image * 255).astype(np.uint8)  # Normalize back to [0, 255]

# XOR-based encryption using the chaos matrix
def apply_chaos_matrix(image, chaos_matrix):
    encrypted_image = np.bitwise_xor(image, chaos_matrix)
    return encrypted_image

# Arnold Cat Map function for scrambling the image
def arnold_cat_map_encrypt(image, k1, iterations):
    """Encrypt the image using Arnold's Cat Map."""
    height, width = image.shape
    new_image = np.copy(image)
    
    for _ in range(iterations):
        transformed_image = np.zeros_like(image)
        for x in range(height):
            for y in range(width):
                new_x = (x + k1 * y) % height
                new_y = (y + k1 * x) % width
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

# Step 1: Apply Arnold Cat Map Encryption
k1 = 3  # Arnold Cat Map coefficient
iterations_arnold = 10
arnold_encrypted_watermark = arnold_cat_map_encrypt(watermark, k1, iterations_arnold)

# Step 2: Apply XOR encryption using the chaos matrix
chaos_matrix = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)  # Random chaos matrix
xor_encrypted_watermark = apply_chaos_matrix(arnold_encrypted_watermark, chaos_matrix)

# Step 3: Apply Logistic Map Encryption to the XOR encrypted watermark
k2 = 3.9  # Control parameter for logistic map
iterations_logistic = 10
logistic_encrypted_watermark = logistic_map_encrypt(xor_encrypted_watermark, k2, iterations_logistic)

# Step 4: Binarize the logistic encrypted watermark
binarized_scrambled_watermark = binarize_image(logistic_encrypted_watermark)
print("Shape of the final binarized watermark:", binarized_scrambled_watermark.shape)

# Convert to PyTorch tensor
binarized_scrambled_watermark_tensor = torch.tensor(binarized_scrambled_watermark, dtype=torch.uint8)
torch.save(binarized_scrambled_watermark_tensor, 'D:/Dataset_medical/Saved_Model/binarized_scrambled_watermark.pt')

# Display the encrypted and scrambled watermark
plt.figure(figsize=(8, 8))
plt.imshow(binarized_scrambled_watermark, cmap='gray')
plt.title('Watermark Encrypted with Arnold Map, XOR Chaos, and Logistic Map')
plt.axis('off')
plt.show()
