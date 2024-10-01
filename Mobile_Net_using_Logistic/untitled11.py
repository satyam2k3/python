# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 02:07:04 2024

@author: HP
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Step 1: Load the encrypted watermark from a file
def load_encrypted_watermark(filepath):
    encrypted_watermark_tensor = torch.load(filepath)
    encrypted_watermark = encrypted_watermark_tensor.numpy()
    return encrypted_watermark

# Step 2: Generate the logistic chaotic sequence using a logistic map
def logistic_map(x0, r, steps):
    chaotic_sequence = []
    x = x0
    for _ in range(steps):
        x = r * x * (1 - x)
        chaotic_sequence.append(x)
    return np.array(chaotic_sequence)

# Step 3: Binarize the chaotic sequence to produce a chaos matrix
def binarize_chaotic_sequence(sequence, height, width):
    sequence_flat = sequence.flatten()
    min_val = np.min(sequence_flat)
    max_val = np.max(sequence_flat)
    normalized_sequence = (sequence_flat - min_val) / (max_val - min_val)
    binary_sequence = np.where(normalized_sequence > 0.5, 1, 0)
    return binary_sequence[:height * width].reshape(height, width)

# Step 4: Apply the inverse Arnold Cat Map to unscramble the image
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

# Step 5: Perform XOR decryption between the unscrambled image and chaos matrix
def apply_chaos_matrix(image, chaos_matrix):
    # Resize chaos matrix to match the size of the image
    if image.shape != chaos_matrix.shape:
        chaos_matrix_resized = cv2.resize(chaos_matrix, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        chaos_matrix_resized = chaos_matrix
    
    # Ensure both image and chaos matrix are integers for XOR
    image = image.astype(np.uint8)
    chaos_matrix_resized = chaos_matrix_resized.astype(np.uint8)

    return np.bitwise_xor(image, chaos_matrix_resized)
# Step 6: Display the decrypted watermark
def display_image(image, title="Decrypted Watermark"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

# Step 7: Save the decrypted watermark if needed
def save_image(image, filepath):
    cv2.imwrite(filepath, image * 255)

# Main decryption function following the outlined steps
def decrypt_watermark(encrypted_image_path, a_arnold, b_arnold, arnold_iters, x0, r, height, width):
    # Step 1: Load the encrypted watermark
    encrypted_watermark = load_encrypted_watermark(encrypted_image_path)

    # Step 2: Generate the logistic chaotic sequence
    steps = height * width
    chaotic_sequence = logistic_map(x0, r, steps)

    # Step 3: Binarize the chaotic sequence to produce the chaos matrix
    chaos_matrix = binarize_chaotic_sequence(chaotic_sequence, height, width)

    # Step 4: Apply the inverse Arnold Cat Map
    unscrambled_watermark = inverse_arnold_cat_map(encrypted_watermark, a_arnold, b_arnold, arnold_iters)

    # Step 5: Perform XOR decryption between the unscrambled image and chaos matrix
    decrypted_watermark = apply_chaos_matrix(unscrambled_watermark, chaos_matrix)

    # Step 6: Display the decrypted watermark
    display_image(decrypted_watermark)

    # Step 7: Save the decrypted watermark (if needed)
    save_image(decrypted_watermark, 'decrypted_watermark.png')
    print("Decrypted watermark saved as 'decrypted_watermark.png'")

# Example usage
encrypted_image_path = "D:/Dataset_medical/Saved_Model/extracted_attack_watermark.pt"  # Filepath for encrypted watermark
a_arnold = 3  # Arnold Cat Map parameter
b_arnold = 5  # Arnold Cat Map parameter
arnold_iters = 10  # Number of iterations for Arnold Cat Map
x0 = 0.5  # Initial condition for logistic map
r = 3.9  # Control parameter for logistic map
height, width = 225, 225  # Example image dimensions

decrypt_watermark(encrypted_image_path, a_arnold, b_arnold, arnold_iters, x0, r, height, width)
