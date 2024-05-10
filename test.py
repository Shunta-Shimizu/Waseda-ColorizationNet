import os
import torch
import torch.nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import glob
import random
import argparse
from tqdm import tqdm
from model import WasedaColorizationNet
from dataset import Gray2RGBDataset
from tools import Lab2RGB

parser = argparse.ArgumentParser()
parser.add_argument("--test_data_dir", type=str, default=None)
parser.add_argument("--model_path", type=str, default="./pretrained_model/ImageNet_ILSVRC2012.pth")
parser.add_argument("--save_result_dir", type=str, default="./output/ImageNet/")
parser.add_argument("--batch_size", type=int, default=1)

config = parser.parse_args()

# model_path = "~/waseda_color/model/Places365_standard256.pth"
# model_path = "~/waseda_color/model/ImageNet/ILSVRC2012.pth"
# model_path = config.model_path
model_path = os.path.expanduser(config.model_path)
# test_data_path = "~/Places/test_256/"
# test_data_path = "~/ImageNet/ILSVRC2012_img_test/"
# test_data_path = os.path.expanduser(test_data_path)
test_img_files = []
test_data_path = config.test_data_dir
test_img_files = glob.glob(test_data_path+"*")

random.seed(22)
random.shuffle(test_img_files)
test_img_files = test_img_files[:1000]

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_dataset = Gray2RGBDataset(transform, test_img_files)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = WasedaColorizationNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

img_size = 224

with tqdm(test_dataloader, total=len(test_dataloader)) as pbar:
    i = 0
    for input_gray_img, input_scale_img, input_L, gt_ab in pbar:
        name = test_img_files[i].split("/")[-1]

        input_gray_img = input_gray_img.to(device)

        torchvision.utils.save_image(input_gray_img, fp="./output/ImageNet/gray_1000/"+name)
        input_scale_img = input_scale_img.to(device)

        output_ab, class_pred = model(input_gray_img, input_scale_img)

        input_gray_img = input_gray_img.squeeze(0)
        output_ab = output_ab.squeeze(0) 
        L = input_gray_img.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
        output_ab = output_ab.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)

        output_img = Lab2RGB(L, output_ab)

        output_img = Image.fromarray(np.clip(output_img*255.0, a_min=0, a_max=255).astype(np.uint8))

        output_img.save(config.save_result_dir+name)
        
        i += 1

