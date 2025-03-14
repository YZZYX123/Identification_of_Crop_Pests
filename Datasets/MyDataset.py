"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
            print("\n  {} has been convert to RGB mode.".format(self.images_path[item]))
            # raise ValueError("image: {} isn't RGB mode".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):  # 整理数据
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

