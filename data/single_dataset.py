import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import sys
import os
import cv2


class SingleDataset(data.Dataset):
    def __init__(self, names, class_labels, domain_labels, dataset_path, img_transformer=None):
        self.data_path = dataset_path
        self.names = names
        self.class_labels = class_labels
        self.domain_labels = domain_labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return cv2.imread(framename), self._image_transformer(img), \
               int(self.class_labels[index] - 1), int(self.domain_labels[index] - 1)

    def __len__(self):
        return len(self.names)


