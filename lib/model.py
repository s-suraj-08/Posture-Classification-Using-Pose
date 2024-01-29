from torch._C import device
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import os

from torchvision.models.detection.keypoint_rcnn import KeypointRCNN

class PoseModel(nn.Module):
    def __init__(self) -> None:
        super(PoseModel,self).__init__()
        self.pose = keypointrcnn_resnet50_fpn(pretrained=True)
    
    def forward(self, x,tar):
        out = self.pose(x,tar)
        return out

