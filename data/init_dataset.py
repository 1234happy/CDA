import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import os
import cv2


def rgb_loader(path):
    return Image.open(path).convert('RGB')
    
class ImageListDataset(data.Dataset):

    def __init__(self, txt_path_rgb, transform=None):
        fh_rgb = open(txt_path_rgb, 'r')
        imgs = []
        for line_rgb in fh_rgb:
            line_rgb = line_rgb.rstrip()
            words_rgb = line_rgb.split()

            imgs.append((words_rgb[0], int(words_rgb[1])))


        self.imgs = imgs
        self.transform = transform
        self.loader=rgb_loader

    def __getitem__(self, index):
        rgb_path, label_rgb= self.imgs[index]       
        img = self.loader(rgb_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label_rgb,rgb_path

    def __len__(self):
        return len(self.imgs)