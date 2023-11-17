import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from modules import LowLevelFeaturesNet, GlobalFeaturesNet, MidLevelFeaturesNet, ClassificationNet, FusionLayer, ColorizationNet, OutputLayer

class WasedaColorizationNet(nn.Module):
    def __init__(self):
        super(WasedaColorizationNet, self).__init__()
        self.low_level_features = LowLevelFeaturesNet(in_channels=1, out_channels=512)
        self.global_features = GlobalFeaturesNet(in_channels=512, out_channels=256)
        self.mid_level_features = MidLevelFeaturesNet(in_channels=512, out_channels=256)
        self.classfication = ClassificationNet(in_channels=512, out_channels=128) # out_channels = num_classes (check: dataset)
        self.fusion = FusionLayer(512, 256)
        self.colorize = ColorizationNet(in_channels=256, out_channels=2, scale_factor=2, mode='nearest')
        self.out = OutputLayer(scale_factor=2, mode='nearest')

    def forward(self, x, x_scale):
        # x_input = x
        # x_scale = x_input.reshape((10, 1, 224, 224))
        x_low = self.low_level_features(x)
        x_lowscale = self.low_level_features(x_scale)

        x_mid = self.mid_level_features(x_low)
        x_global, x_class_in = self.global_features(x_lowscale)

        x_class_pred = self.classfication(x_class_in)

        x_fusion = self.fusion(x_mid, x_global)

        x_color= self.colorize(x_fusion)

        x_out = self.out(x_color)

        return x_out, x_class_pred

