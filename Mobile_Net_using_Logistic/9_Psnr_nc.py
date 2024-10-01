# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:31:18 2024

@author: HP

FINAL STEP: This code is used to calculate the psnr and nc values of the original watermark and extracted watermark from the attacked image
"""

import numpy as np
import cv2

def calculate_psnr(original_image, attacked_image):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between the original and attacked images.
    
    Parameters:
    - original_image: The original image
    - attacked_image: The attacked image
    
    Returns:
    - psnr_value: The PSNR value in decibels
    """
    mse = np.mean((original_image - attacked_image) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if there's no difference
    
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_nc(original_image, attacked_image):
    """
    Calculate Normalized Correlation (NC) between the original and attacked images.
    
    Parameters:
    - original_image: The original image
    - attacked_image: The attacked image
    
    Returns:
    - nc_value: The NC value
    """
    original_image = original_image.astype(float)
    attacked_image = attacked_image.astype(float)
    
    mean_original = np.mean(original_image)
    mean_attacked = np.mean(attacked_image)
    
    numerator = np.sum((original_image - mean_original) * (attacked_image - mean_attacked))
    denominator = np.sqrt(np.sum((original_image - mean_original) ** 2) * np.sum((attacked_image - mean_attacked) ** 2))
    
    if denominator == 0:
        return 0  # To avoid division by zero
    
    nc_value = numerator / denominator
    return nc_value

def process_and_evaluate(original_image_path, attacked_image_path):
    """
    Load images, calculate PSNR and NC, and print the results.
    
    Parameters:
    - original_image_path: Path to the original image
    - attacked_image_path: Path to the attacked image
    """
    # Load the images
    original_image = cv2.imread(original_image_path)
    attacked_image = cv2.imread(attacked_image_path)
    
    if original_image is None or attacked_image is None:
        raise FileNotFoundError("Could not load one or both images.")
    
    # Convert images to grayscale if they are color images
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(attacked_image.shape) == 3:
        attacked_image = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2GRAY)
    
    # Resize the attacked image to match the original image's dimensions
    attacked_image = cv2.resize(attacked_image, (original_image.shape[1], original_image.shape[0]))
    
    # Calculate PSNR and NC
    psnr_value = calculate_psnr(original_image, attacked_image)
    nc_value = calculate_nc(original_image, attacked_image)
    
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"Normalized Correlation: {nc_value:.2f}")

# Example usage
original_image_path ='C:/Users/HP/OneDrive/Desktop/images.png'  # Path to the original imageinput_image_path = '/kaggle/input/coviddc3/DC1test20/DC1test20/control/segm-controlNovara1-1.png'

attacked_image_path = 'C:/Users/HP/.spyder-py3/Mobile_Net_using_Logistic/decrypted_watermark.png'  # Path to the attacked image

process_and_evaluate(original_image_path, attacked_image_path)
# try:
#     process_and_evaluate(original_image_path, attacked_image_path)
# except Exception as e:
#     print(f"An error occurred: {e}")
