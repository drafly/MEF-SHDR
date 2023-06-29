import os
import math
import numpy as np
from glob import glob
import copy
from random import shuffle
from PIL import Image, ImageFilter

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import cv2


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.image_path = []
        for ext in ['*.jpg', '*_A.png']: 
            self.image_path.extend(glob(os.path.join(args.dir_image, ext)))
        self.mask_path = glob(os.path.join(args.dir_image, '*_T.png'))
        self.mask_len = len(self.mask_path)

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor()])
        self.mask_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(
                (0, 45), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        HSV_img = image.convert("HSV")

        gt_image = Image.open(self.image_path[index].replace('_A.png', '_D.png')).convert('RGB')

        highlight_mask_path = self.image_path[index].replace('_A.png', '_T.png')
        highlight_mask = Image.open(highlight_mask_path).convert('L')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        HSV_img = self.img_trans(HSV_img) 
        HSV_img = HSV_img * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        
        torch.random.manual_seed(seed)
        highlight_mask = self.mask_trans(highlight_mask)
        highlight_mask = np.array(highlight_mask)
        highlight_mask = torch.tensor(highlight_mask)
        highlight_mask = torch.unsqueeze(highlight_mask, dim=0)
        highlight_mask = highlight_mask / 255.

        return image, gt_image, highlight_mask, HSV_img, filename

class InpaintingDataTest(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        # image and mask 
        self.image_path = []
        for ext in ['*.jpg', '*_A.png']: 
            self.image_path.extend(glob(os.path.join(args.dir_test, ext)))
        self.mask_path = glob(os.path.join(args.dir_test, '*_T.png'))
        self.mask_len = len(self.mask_path)

        # augmentation 
        self.img_trans = transforms.Compose([
            transforms.ToTensor()])

        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])
        HSV_img = image.convert("HSV")

        gt_image = Image.open(self.image_path[index].replace('_A.png', '_D.png')).convert('RGB')

        highlight_mask_path = self.image_path[index].replace('_A.png', '_T.png')
        highlight_mask = Image.open(highlight_mask_path).convert('L')

        # augment
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = self.img_trans(image) 
        image = image * 2. - 1.

        torch.random.manual_seed(seed)
        HSV_img = self.img_trans(HSV_img) 
        HSV_img = HSV_img * 2. - 1.

        torch.random.manual_seed(seed)
        gt_image = self.img_trans(gt_image) # ([1, 512, 512])
        gt_image = gt_image * 2. - 1.
        
        torch.random.manual_seed(seed)
        highlight_mask = np.array(highlight_mask)
        highlight_mask = torch.tensor(highlight_mask)
        highlight_mask = torch.unsqueeze(highlight_mask, dim=0)
        highlight_mask = highlight_mask / 255.

        return image, gt_image, highlight_mask, HSV_img, filename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--dir_image', default='./dataset/SHIQ_data_10825/train', type=str)
    parser.add_argument('--image_size', default='200', type=int)
    args = parser.parse_args()

    data = InpaintingData(args)
    print(len(data))
    img, highlight, hsi, filename = data[10]
    from  torchvision import utils as vutils
    vutils.save_image(img, './image.jpg', normalize=True)

    highlight = highlight.type_as(img)
    vutils.save_image(highlight, './hmask.jpg', normalize=True)
    print(img.size(), highlight.size(), filename)
