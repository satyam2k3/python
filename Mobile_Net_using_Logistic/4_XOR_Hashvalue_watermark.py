# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 02:50:51 2024

@author: HP

STEP 4 : doing XOR of HASHING VALUES and BINARIZED DISLOCATED WATERMARK (this is generating the KEY values)
    
This is the third file doing XOR of HASHING VALUES and BINARIZED DISLOCATED WATERMARK
THE RESULT GENRATED IS THE KEY ONLY 
"""
import torch
import numpy as np

# Load the data (removing weights_only=True as it's not needed)
load_hashing_values = torch.load("D:/Dataset_medical/Saved_Model/hashing_values.pt")
load_dislocated_watermark = torch.load('D:/Dataset_medical/Saved_Model/binarized_scrambled_watermark.pt')

# If the loaded data is a NumPy array, convert it to a PyTorch tensor
if isinstance(load_hashing_values, np.ndarray):
    load_hashing_values = torch.from_numpy(load_hashing_values)

if isinstance(load_dislocated_watermark, np.ndarray):
    load_dislocated_watermark = torch.from_numpy(load_dislocated_watermark)

# Ensure the data types are compatible for XOR operation
# Convert both tensors to the same dtype if necessary
load_hashing_values = load_hashing_values.to(torch.uint8)
load_dislocated_watermark = load_dislocated_watermark.to(torch.uint8)

# Get dimensions
hashing_values_size = load_hashing_values.size()
dislocated_watermark_size = load_dislocated_watermark.size()

# Reshape or repeat load_hashing_values to match the dimensions of load_dislocated_watermark
tensor_a = load_hashing_values
tensor_b = load_dislocated_watermark

# Calculate the number of elements in tensor_a to match tensor_b dimensions
repeat_x = tensor_b.size(0) // tensor_a.size(0) + 1
repeat_y = tensor_b.size(1) // tensor_a.size(1) + 1

# Repeat and reshape tensor_a to match the dimensions of tensor_b
tensor_a_repeated = tensor_a.repeat(repeat_x, repeat_y)[:tensor_b.size(0), :tensor_b.size(1)]

# Perform the XOR operation
result_tensor = torch.bitwise_xor(tensor_b, tensor_a_repeated)

# Save the binarized dislocated watermark as a .pt file
torch.save(result_tensor, 'D:/Dataset_medical/Saved_Model/result_tensor.pt')

# Print the result
print("Result Tensor Shape:", result_tensor.shape)
print("Result Tensor:", result_tensor)
print(type(result_tensor))
