# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 18:55:13 2024

@author: HP

USING CREATE AND LOAD DATASET (NEW)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# Create a Custom Dataset Class

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert masks to tensor
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # Instances are encoded as different colors
        obj_ids = torch.unique(mask)[1:]  # Exclude background
        masks = mask == obj_ids[:, None, None]

        # Get bounding boxes for each object
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = torch.where(masks[i])
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels (e.g., 1 for one class, 2 for another, etc.)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # Adjust based on your classes

        # Image ID
        image_id = torch.tensor([idx])

        # Areas of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # No crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)




# Define Data Transformations

import torchvision.transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



# Load the Dataset

# Root directory of the dataset
dataset = CustomDataset('D:/Dataset_drone/huggingface_drone_detection_2gb/data/train', transforms=get_transform(train=True))
dataset_test = CustomDataset('D:/Dataset_drone/huggingface_drone_detection_2gb/data/test', transforms=get_transform(train=False))

# Split the dataset into training and validation sets
indices = torch.randperm(len(dataset)).tolist()
train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x))
)
test_data_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x))
)






















