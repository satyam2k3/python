# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 04:30:34 2024

@author: HP
For loading dataset in Fast RCNN 
"""

import torch
import torchvision

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)

# Modify the head of the network to fit your dataset
num_classes = 21  # Example for 20 classes + background for Pascal VOC
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
 

from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None):
        self.images_dir = images_dir
        self.annotations = json.load(open(annotations_file, 'r'))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.annotations[idx]['image'])
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor(self.annotations[idx]['boxes'], dtype=torch.float32)
        labels = torch.tensor(self.annotations[idx]['labels'], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}

        return image, target


from torchvision import transforms

# Define any transformations (e.g., resizing, normalizing)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize your custom dataset
dataset = CustomDataset(images_dir='D:/Dataset_drone/huggingface_drone_detection_2gb/data/test/images/test0002.jpg', annotations_file='D:/Dataset_drone/huggingface_drone_detection_2gb/data/test/labels/test0002.txt', transform=transform)

# Create a DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
