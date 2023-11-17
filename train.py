import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataset import Gray2RGBDataset
from model import WasedaColorizationNet

train_img_files = []
val_img_files = [] 

# Places365
# train_data_path = "/home/shimizu/Places/data_256/"
# val_data_path = "/home/shimizu/Places/val_256/"

# ImageNet
train_data_path = "/home/shimizu/ImageNet/ILSVRC2012_img_train/"
val_data_path = "/home/shimizu/ImageNet/ILSVRC2012_img_val/"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip(p=0.5)])

for file in glob.glob(train_data_path+"**", recursive=True):
    if os.path.isfile(file) and file.split("/")[-1].split(".")[-1] == "JPEG":
        train_img_files.append(file)

for file in glob.glob(val_data_path+"*"):
    if os.path.isfile(file):
        val_img_files.append(file)
# np.random.seed(0)
# np.random.shuffle(img_files)
# split_n = int(len(img_files)*0.9)
# train_img_files = img_files[:split_n]
# val_img_files = img_files[split_n:]

train_dataset = Gray2RGBDataset(transform, train_img_files)
val_dataset = Gray2RGBDataset(transform, val_img_files)

batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = WasedaColorizationNet()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)
output_criterion = nn.MSELoss()
class_criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.Adadelta(model.parameters())
# print(len(test_dataloader), len(test_dataloader.dataset))

loss_history = dict()
loss_history["train"] = []
loss_history["val"] = []
epochs = 22
for i in range(epochs):
    print("epoch{}:".format(str(i+1)))

    model.train()
    train_epoch_loss = 0
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for input_gray_img, input_scale_img, input_L, gt_ab in pbar:
            input_gray_img = input_gray_img.to(device)
            input_scale_img = input_scale_img.to(device)
            gt_ab = gt_ab.to(device)
            optimizer.zero_grad()
            output_ab, class_pred = model(input_gray_img, input_scale_img)
            loss = output_criterion(output_ab, gt_ab)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            train_epoch_loss += batch_loss

    train_epoch_loss /= len(train_dataloader)
    loss_history["train"].append(train_epoch_loss)
    print("Train mean loss epoch{} = {}".format(i+1, train_epoch_loss))

    model.eval()
    val_epoch_loss = 0
    for input_gray_img, input_scale_img, input_L, gt_ab in val_dataloader:
        input_gray_img = input_gray_img.to(device)
        input_scale_img = input_scale_img.to(device)
        gt_ab = gt_ab.to(device)
        # optimizer.zero_grad()
        output_ab, class_pred = model(input_gray_img, input_scale_img)
        loss = output_criterion(output_ab, gt_ab)
        batch_loss = loss.item()
        val_epoch_loss += batch_loss
    
    val_epoch_loss /= len(val_dataloader)
    loss_history["val"].append(val_epoch_loss)
    print("Validation mean loss epoch{} = {}".format(i+1, val_epoch_loss))

    if i == 0:
        min_loss = val_epoch_loss
        # min_loss = sum_loss*batch_size / len(val_dataloader.dataset)
        # min_loss = sum_loss / len(test_dataloader)
    else:
        # loss_i = sum_loss * batch_size / len(test_loader.dataset)
        # loss_i = sum_loss*batch_size / len(val_dataloader.dataset)
        loss_i = val_epoch_loss
        if loss_i < min_loss:
            min_loss = loss_i
            model = model.to("cpu")
            torch.save(model.module.state_dict(), "/home/shimizu/waseda_color/model/ImageNet/ILSVRC2012.pth")
            print("save model")
            model = model.to(device)
        elif (i+1)%10 == 0:
            model = model.to("cpu")
            torch.save(model.module.state_dict(), "/home/shimizu/waseda_color/model/ImageNet/ILSVRC2012_epoch{}.pth".format(str(i+1)))
            print("save model")
            model = model.to(device)
