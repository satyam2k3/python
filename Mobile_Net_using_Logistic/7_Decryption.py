# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:06:35 2024

@author: HP


This is just a step which is verifying that XOR with hashing values and key values and verifying it with the disloacted watermark values

THIS IS NOT REALLY NECESSARY FILE
"""

import torch
import matplotlib.pyplot as plt

# Load the data
load_hashing_values = torch.load("D:/Dataset_medical/Saved_Model/attacked_hash_values.pt", weights_only=True) # temp 
print("Hashing values shape:", load_hashing_values.shape)

key_values = torch.load("D:/Dataset_medical/Saved_Model/result_tensor.pt", weights_only=True) # XOR Hashvalue watermark (key) 
print("Key values shape:", key_values.shape)

load_dislocated_watermark = torch.load('D:/Dataset_medical/Saved_Model/binarized_scrambled_watermark.pt', weights_only=True) #Encryption _grey
print("Dislocated watermark shape:", load_dislocated_watermark.shape)

# Ensure the data types are compatible for XOR operation
# Convert both tensors to the same dtype if necessary
load_hashing_values = load_hashing_values.to(torch.uint8)
key_values = key_values.to(torch.uint8)


# Get dimensions
hashing_values_size = load_hashing_values.size()
dislocated_watermark_size = key_values.size()


# Reshape or repeat load_hashing_values to match the dimensions of load_dislocated_watermark
# Assuming you want to repeat the hashing values to match the size of dislocated watermark
tensor_a = load_hashing_values
tensor_b = key_values


# Calculate the number of elements in tensor_a to match tensor_b dimensions
# Expand tensor_a to match the width of tensor_b
repeat_x = tensor_b.size(0) // tensor_a.size(0) + 1 
repeat_y = tensor_b.size(1) // tensor_a.size(1) + 1



# Repeat and reshape tensor_a to match the dimensions of tensor_b
tensor_a_repeated = tensor_a.repeat(repeat_x, repeat_y)[:tensor_b.size(0), :tensor_b.size(1)]

# Perform the XOR operation
result_tensor = torch.bitwise_xor(tensor_b, tensor_a_repeated)

# Save the binarized dislocated watermark as a .pt file
torch.save(result_tensor, 'D:/Dataset_medical/Saved_Model/extracted_attack_watermark.pt')

# Print the result
print("Result Tensor Shape:", result_tensor.shape)
print("Result Tensor:", result_tensor)
print(type(result_tensor))

are_equal = torch.equal(result_tensor, load_dislocated_watermark)
print(are_equal)


# Pixel-wise accuracy
num_matching_elements = torch.sum(result_tensor == load_dislocated_watermark).item()
total_elements = result_tensor.numel()

accuracy = num_matching_elements / total_elements * 100  # Percentage of similarity
print(f"Pixel-wise Accuracy: {accuracy:.2f}%")


plt.figure(figsize=(6, 6))
plt.imshow(result_tensor.cpu().numpy(), cmap='gray')  # Convert to NumPy array for plotting
plt.title("Binarized Dislocated Watermark After XOR")
plt.axis('off')  # Turn off axis
plt.show()