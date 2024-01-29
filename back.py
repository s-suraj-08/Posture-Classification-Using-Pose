from typing import List
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, dataloader


from pathlib import Path
import numpy as np
import random
import argparse

from lib.data_set import WLDataset

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


GOOD = 1
BAD = 0
USE_GPU = True
learning_rate = 0.2
learning_rate_decay = 0.99
train_dir = './data/train'
num_workers = 1
num_epochs = 10

if torch.cuda.is_available()and USE_GPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# create the list of keypoints.
keypoints = ['nose','left_eye','right_eye',\
'left_ear','right_ear','left_shoulder',\
'right_shoulder','left_elbow','right_elbow',\
'left_wrist','right_wrist','left_hip',\
'right_hip','left_knee', 'right_knee', \
'left_ankle','right_ankle']

def get_train_valid(files,num_workers, validation_split=0.2):
    random.shuffle(files)
    split_index = int((1 - validation_split) * len(files))
    train_data = files[:split_index]
    valid_data = files[split_index:]
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = WLDataset(train_data, transform)
    valid_dataset = WLDataset(valid_data, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=2, num_workers=num_workers, shuffle=True)

    val_loader = DataLoader(valid_dataset, batch_size=2,
                            num_workers=num_workers, shuffle=True)
    return train_loader, val_loader

##loading image paths
train_files = list(Path(train_dir).glob('**/*.png'))
train_loader, val_loader = get_train_valid(train_files,num_workers)

class PoseModel(nn.Module):
    def __init__(self, pretrained=True):
        super(PoseModel,self).__init__()
        self.pose = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    
    def forward(self, x):
        out = self.pose(x)
        return out

model = PoseModel()
params_to_update = model.parameters
model.to(device)

#loss and optimiser
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#to get accuracy
max_acc = 0
acc_percentage = []

for epoch in range(num_epochs):
    for image,labels in enumerate(train_loader):
        print(image)