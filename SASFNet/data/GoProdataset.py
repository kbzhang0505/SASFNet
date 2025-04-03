import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random
import torchvision.transforms.functional as TF

class GoProDataset(Dataset):
    def __init__(self, blur, sharp, edge, crop=True, crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=True):
        '''
        with open(blur_image_files, "r") as f:
            data = f.readlines()
            self.blur_image_files = data[0:2028]
            self.sharp_image_files = data[2028:4056]
        '''
        with open(blur, "r") as f:
            self.blur_dir = f.readlines()
        with open(sharp, "r") as d:
            self.sharp_dir = d.readlines()
        with open(edge, "r") as g:
            self.edge_dir = g.readlines()
        self.transform = transform
        self.transform_tensor = transforms.ToTensor()
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)
        self.rotate45 = transforms.RandomRotation(45)

    def __len__(self):
        return len(self.blur_dir)

    def name(self):
        return 'GoProData'

    def __getitem__(self, idx):
        image_name = self.blur_dir[idx].split('/')[-1]
        blur_image = Image.open(self.blur_dir[idx].rstrip()).convert('RGB')
        sharp_image = Image.open(self.sharp_dir[idx].rstrip()).convert('RGB')
        edge_image = Image.open(self.edge_dir[idx].rstrip()).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree)
            sharp_image = transforms.functional.rotate(sharp_image, degree)
            edge_image = transforms.functional.rotate(edge_image, degree)

        if self.color_augment:
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            edge_image = transforms.functional.adjust_gamma(edge_image, 1)
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)
            edge_image = transforms.functional.adjust_saturation(edge_image, sat_factor)

        if self.transform:
            blur_image = self.transform_tensor(blur_image)
            sharp_image = self.transform_tensor(sharp_image)
            edge_image = self.transform_tensor(edge_image)
        
        if self.crop:
            w = blur_image.size()
            W = w[1]
            h = blur_image.size()
            H = h[2]

            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]

            blur_image = blur_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            sharp_image = sharp_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            edge_image = edge_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
        
        return {'A': blur_image, 'B': sharp_image, 'e': edge_image, 'A_paths': self.blur_dir[idx].rstrip(), 'B_paths': self.sharp_dir[idx].rstrip()}


class GoProDataset_test(Dataset):
    def __init__(self, blur, sharp, crop=False, crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=True):
        with open(blur, "r") as f:
            self.blur_dir = f.readlines()
        with open(sharp, "r") as d:
            self.sharp_dir = d.readlines()
        self.transform = transform
        self.transform_tensor = transforms.ToTensor()
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)
        self.rotate45 = transforms.RandomRotation(45)

    def __len__(self):
        return len(self.blur_dir)

    def name(self):
        return 'GoProData_test'

    def __getitem__(self, idx):
        image_name = self.blur_dir[idx][0:-1].split('/')
        blur_image = Image.open(self.blur_dir[idx].rstrip()).convert('RGB')
        sharp_image = Image.open(self.sharp_dir[idx].rstrip()).convert('RGB')
        # blur_image = blur_image.resize((668, 756), 3)
        # sharp_image = sharp_image.resize((668, 756), 3)
        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree)
            sharp_image = transforms.functional.rotate(sharp_image, degree)

        if self.color_augment:
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)

        if self.transform:
            blur_image = self.transform_tensor(blur_image)
            sharp_image = self.transform_tensor(sharp_image)
        
        if self.crop:
            w = blur_image.size()
            W = w[1]
            h = blur_image.size()
            H = h[2]

            Ws = np.random.randint(0, W - self.crop_size - 1, 1)[0]
            Hs = np.random.randint(0, H - self.crop_size - 1, 1)[0]

            blur_image = blur_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
            sharp_image = sharp_image[:, Ws:Ws + self.crop_size, Hs:Hs + self.crop_size]
        
        return {'A': blur_image, 'B': sharp_image, 'A_paths': self.blur_dir[idx].rstrip(), 'B_paths': self.sharp_dir[idx].rstrip()}
