import torch
from torch.utils.data import Dataset
from torchvision import transforms,  io
from typing import TypedDict, List
from pathlib import Path
import json
import numpy as np

import cv2

GOOD = 1
BAD = 0



class Sample(TypedDict):
    label: str
    image: torch.Tensor

class WLDataset(Dataset):
    def __init__(self, files: List[Path], transform: transforms.Compose = None, 
                read_keypoints: bool = True, show_keypoints: bool = False) -> None:
        super().__init__()
        self.files = files
        self.transform = transform
        self.read_keypoints = read_keypoints
        self.show_keypoints = show_keypoints

    def get_pose_label(self, file: Path) -> str:
        return GOOD if 'good' in file.stem  else BAD

    def __len__(self) -> int:
        return len(self.files)


    def transform_image(self, image):
        image = cv2.resize(image, (640, 360))
        return image

    def __getitem__(self, index: int) -> Sample:
        file = self.files[index]
        assert file.exists(), f'File does not exist: {file}'
        image = cv2.imread(str(file))

        image = self.transform_image(image)
        
        if self.transform:
            image = self.transform(image)

        label = self.get_pose_label(file)

        keypoint_data = {}
        if self.read_keypoints:
            count = 0
            with file.with_name(file.stem + '_keypoints.json').open() as file:
                _data = json.load(file)
                keypoint_data['boxes'] = np.array(_data['boxes'])
                keypoint_data['keypoints'] = np.array(_data['keypoints'])
                keypoint_data['labels'] = np.array(_data['labels'])
                keypoint_data['image_id'] = count
                count = count + 1
        data = {'label': torch.tensor(label), 'image': image, 'file': str(file),'iscrowd':False}
        data.update(keypoint_data)
        return data


