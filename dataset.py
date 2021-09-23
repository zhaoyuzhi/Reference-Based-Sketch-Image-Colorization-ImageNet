import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset

import utils
from util.transformation_tps import tps_transform
from util.transformation_elastic import elastic_transform

class ImageNet_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = []

        # Build training dataset
        imglist = utils.get_jpgs(opt.baseroot_train)
        for i in range(len(imglist)):
            self.imglist.append(os.path.join(opt.baseroot_train, imglist[i]))

    def __getitem__(self, index):

        # Path of images
        img_path = self.imglist[index]

        # Read images
        img = cv2.imread(img_path)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Build reference image
        # add noise
        if self.opt.noise_type == 'uniform':
            noise = np.random.uniform(self.opt.a, self.opt.b, img.shape)
        elif self.opt.noise_type == 'gaussian':
            noise = np.random.normal(self.opt.mean, self.opt.std, img.shape)
        ref = img + noise
        ref = np.clip(ref, 0, 255).astype(np.uint8)
        # apply transformation
        if self.opt.trans_type == 'elastic':
            ref = elastic_transform(ref, 1000, 8, random_state = None)
        elif self.opt.trans_type == 'tps':
            ref = tps_transform(ref)

        # Process images
        grayimg = grayimg.astype(np.float) / 255.0
        grayimg = torch.from_numpy(grayimg).float().unsqueeze(0).contiguous()
        ref = ref.astype(np.float) / 255.0
        ref = torch.from_numpy(ref).float().permute(2, 0, 1).contiguous()
        img = img.astype(np.float) / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1).contiguous()

        return grayimg, ref, img
    
    def __len__(self):
        return len(self.imglist)

class ImageNet_ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = []

        # Build training dataset
        self.imglist = utils.text_readlines('util/ctest10k.txt')

    def __getitem__(self, index):
        
        # Path of images
        img_path = self.imglist[index]
        full_path = os.path.join(self.opt.baseroot_val, img_path)
        ref_path = os.path.join(self.opt.baseroot_ref, img_path)

        # Read images
        img = cv2.imread(full_path)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref = cv2.imread(ref_path)

        # Process images
        grayimg = grayimg.astype(np.float) / 255.0
        grayimg = torch.from_numpy(grayimg).float().unsqueeze(0).contiguous()
        ref = ref.astype(np.float) / 255.0
        ref = torch.from_numpy(ref).float().unsqueeze(0).contiguous()
        img = img.astype(np.float) / 255.0
        img = torch.from_numpy(img).float().permute(2, 0, 1).contiguous()

        return grayimg, ref, img, img_path
    
    def __len__(self):
        return len(self.imglist)
