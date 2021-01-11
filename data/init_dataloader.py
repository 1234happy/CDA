from .init_dataset import ImageListDataset
import torch.utils.data
import os
from torchvision import transforms
import numpy as np
import cv2

def get_loader(args):
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])     

    print(args.data_path_test)
    dataset = ImageListDataset(txt_path_rgb=args.data_path_test, transform=test_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False,
                                                 num_workers = 1, pin_memory=True, drop_last = False)
    return dataset_loader


