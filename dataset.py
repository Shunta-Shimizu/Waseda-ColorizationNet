import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import glob
from tools import RGB2ab

class Gray2RGBDataset(Dataset):
    def __init__(self, transform, img_files):
        self.transform = transform
        self.img_files = img_files

    def __getitem__(self, index):
        name = self.img_files[index].split("/")[-1]
        img_size = 224
        # input_gray_img = Image.open(self.img_files[index]).convert("L")
        # gt_img = Image.open(self.img_files[index]).convert("RGB")

        # input_gray_img = input_gray_img.resize((img_size, img_size))
        # input_scale_img = input_gray_img.resize((224, 224))
        # gt_img = gt_img.resize((img_size, img_size))

        # input_gray_img = np.array(input_gray_img, dtype=np.float32)
        # input_gray_img = np.array(input_gray_img)[:, :, np.newaxis]

        # # input_scale_img = np.array(input_scale_img) 
        # input_scale_img = np.array(input_scale_img, dtype=np.float32)
        # input_scale_img = np.array(input_scale_img)[:, :, np.newaxis]

        # input_L = input_gray_img
        # gt_ab = RGB2ab(gt_img, use_skimage=True)
        # gt_ab = np.array(gt_ab, dtype=np.float32)

        # input_gray_img = self.transform(input_gray_img)
        # input_scale_img = self.transform(input_scale_img)
        # input_L = self.transform(input_L)
        # gt_ab = self.transform(gt_ab)

        # return input_gray_img, input_scale_img, input_L, gt_ab

        input_gray_img = Image.open(self.img_files[index]).convert("L")
        gt_img = Image.open(self.img_files[index]).convert("RGB")

        input_gray_img = input_gray_img.resize((img_size, img_size))
        input_scale_img = input_gray_img.resize((224, 224))
        gt_img = gt_img.resize((img_size, img_size))

        input_gray_img = np.array(input_gray_img, dtype=np.uint8)
        input_gray_img = np.array(input_gray_img)[:, :, np.newaxis]

        # input_scale_img = np.array(input_scale_img) 
        input_scale_img = np.array(input_scale_img, dtype=np.uint8)
        input_scale_img = np.array(input_scale_img)[:, :, np.newaxis]

        input_L = input_gray_img
        gt_ab = RGB2ab(gt_img, use_skimage=True)
        gt_ab = np.array(gt_ab, dtype=np.float32)

        input_gray_img = self.transform(input_gray_img)
        input_scale_img = self.transform(input_scale_img)
        input_L = self.transform(input_L)
        gt_ab = self.transform(gt_ab)

        return input_gray_img, input_scale_img, input_L, gt_ab

    def __len__(self):
        return len(self.img_files)
