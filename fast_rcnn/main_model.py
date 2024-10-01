# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:52:25 2024

@author: HP
"""

import torch
from torch.utils.data import DataLoader
import dataset_loader  # Importing the dataset and transformation functions

# Define paths to your dataset
train_data_path = 'D:/Dataset_drone/huggingface_drone_detection_2gb/data/train'
test_data_path = 'D:/Dataset_drone/huggingface_drone_detection_2gb/data/test'

# Initialize datasets
dataset = dataset_loader.CustomDataset(train_data_path, transforms=dataset_loader.get_transform(train=True))
dataset_test = dataset_loader.CustomDataset(test_data_path, transforms=dataset_loader.get_transform(train=False))



# import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.optim as optim
# from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate  # Assuming you have an engine.py file with these functions
import utils  # Assuming you have a utils.py file for utility functions like collate_fn
import torchvision.transforms as T  # Assuming transforms are defined here
# import dataset_loader  # Your custom dataset loader


# LOAD THE PRETRAINED MODEL

 
# Load the pre-trained Mask R-CNN model with ResNet50 backbone
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)

# Replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (drone) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Replace the mask predictor with a new one (if you need mask predictions)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

print("pretrained model loaded")


# PREPARE DATA LOADERS
def collate_fn(batch):
    return tuple(zip(*batch))


train_data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn
)

test_data_loader = DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
)

print("data loaders are fine and working")

# DEFINE THE OPTIMIZER

# Use the SGD optimizer with momentum
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print("SDG optimizer is ready")

# FINE-TUNING THE MODEL

# Move the model to the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Number of epochs
num_epochs = 2




# Your existing training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    
    # Training step
    model.train()
    for images, targets in train_data_loader:
        # Move images and targets to the GPU
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # (Optional) Verify the device of the data
        print(f"Image device: {images[0].device}")
        print(f"Target device: {targets[0]['boxes'].device}")  # Example of checking a tensor in targets
        
        # Now proceed with training as usual
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    # Update the learning rate
    lr_scheduler.step()
    
    # Evaluation step
    eval_results = evaluate(model, test_data_loader, device=device)
    print(f"Evaluation results for epoch {epoch + 1}: {eval_results}")
    
    print()  # Add a blank line between epochs for readability




# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")
#     print("-" * 30)
    
#     # Training step
#     train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    
#     # Update the learning rate
#     lr_scheduler.step()
    
#     # Evaluation step
#     eval_results = evaluate(model, test_data_loader, device=device)
    
#     # Assuming `evaluate` returns some results like accuracy, loss, etc.
#     print(f"Evaluation results for epoch {epoch + 1}: {eval_results}")
    
#     print()  # Add a blank line between epochs for readability


# Save the model after training
torch.save(model.state_dict(), 'mask_rcnn_resnet50_fpn_v2_finetuned.pth')

print("finetuning is done and model is saved at the specified path")




















