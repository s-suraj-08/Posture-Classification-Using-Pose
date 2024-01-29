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
from lib.model import PoseModel


GOOD = 1
BAD = 0
USE_GPU = True
learning_rate = 0.2
learning_rate_decay = 0.99
train_dir = './data/train'
num_workers = 10
num_epochs = 10
IMG_SHAPE = (360, 640, 3)



lr = learning_rate

if torch.cuda.is_available()and USE_GPU:
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

print("Using device:", device)



def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(model: nn.Module, optimiser: torch.optim.Adam,
                    loss: float, epoch: int):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'loss': loss
    }, Path('models', f'checkpoint-{epoch}.cpkt'))


def get_train_valid(files: List[Path], num_workers: int, validation_split=0.2):
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


def run(dataloader: DataLoader, pose_model: nn.Module,
        optimiser: torch.optim.Adam, criterion: nn.MSELoss, train=True):
    correct = 0
    total_loss = 0
    total = 0

    for data in dataloader:
        images, boxes, labels,image_id,keypoints = data['image'],data['boxes'], data['labels'], data['image_id'], data['keypoints']
        images = images.to(device).float()

        boxes = boxes.to(device)
        if len(boxes.shape) > 1:
            boxes = boxes.resize(boxes.shape[0],4)
        else:
            boxes = boxes.resize(1,4)

        image_id = image_id.to(device)

        labels = labels.to(device).int()
        #labels = labels.reshape((labels.shape[0], 1))
        keypoints = keypoints.to(device).float()

        tar = [{'boxes': boxes[i].resize(1,4), 
                 'labels': torch.Tensor(int(labels[i])).type(torch.int64),
                 'keypoints':keypoints[i]} for i in range(len(images))]
        
        
        #keypoints_data = pose_model(images)
        outputs = pose_model(images,tar)
        print(outputs)
        loss = criterion(outputs['loss_keypoint'], keypoints)
        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        total_loss += loss.item()
        total += labels.size(0)
        
        return total_loss / total


def train(train_files: List[Path], num_workers: int, checkpoint: str = None):
    train_loader, val_loader = get_train_valid(train_files, num_workers)

    model = PoseModel()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []
    lr = learning_rate

    for epoch in range(num_epochs):
        train_loss = run(train_loader, model, optimizer, criterion)
        training_losses.append(train_loss)
        print('Training loss is: {} %'.format(train_loss))

        lr *= learning_rate_decay
        update_lr(optimizer, lr)
        
        save_checkpoint(model, optimizer, train_loss, epoch)






def main(args):
    train_files = list(Path(train_dir).glob('**/*.png'))
    #test_files = list(Path(TEST_DATA_DIR).glob('**/*.png'))
    transform = transforms.Compose([transforms.Resize((360, 640))])
    train(train_files, args.cpus, args.checkpoint)

    if args.checkpoint:
        train(train_files, args.cpus, args.checkpoint)
    else:
        # show_sample_images(train_dataset, args.cpus)
        train(train_files, args.cpus)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Pose classifier')
    parser.add_argument('--cpus', dest='cpus', default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default=None, type=str)
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Run test')

    args = parser.parse_args()
    main(args)
