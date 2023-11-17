import torch
import torch.nn as nn
import torchvision

class LowLevelFeaturesNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LowLevelFeaturesNet, self).__init__()
        self.low_features_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels//8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(), 
            nn.Conv2d(out_channels//8, out_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )

        self.low_features_block2 = nn.Sequential(
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )

        self.low_features_block3 = nn.Sequential(
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # print(x.size())
        x = self.low_features_block1(x)
        # print(x.size())
        x = self.low_features_block2(x)
        # print(x.size())
        x = self.low_features_block3(x)
        # print(x.size())
        return x


class MidLevelFeaturesNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidLevelFeaturesNet, self).__init__()
        self.mid_features_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() 
        )
    
    def forward(self, x):
        x = self.mid_features_block(x)

        return x
    
class GlobalFeaturesNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalFeaturesNet, self).__init__()
        self.global_featrues_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.global_featrues_block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.global_featrues_block3 = nn.Sequential(
            nn.Linear(in_channels*7*7, in_channels*2),
            nn.ReLU(),
            nn.Linear(in_channels*2, in_channels),
            nn.ReLU()
        )
        
        self.global_featrues_block4 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.global_featrues_block1(x)
        # print(x.size())
        x = self.global_featrues_block2(x)
        # print(x.size())
        x = x.view(-1, 7*7*512)
        x = self.global_featrues_block3(x)
        x_class = x
        x = self.global_featrues_block4(x)

        return x, x_class

class ClassificationNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        # out_channels = num_classes
        super(ClassificationNet, self).__init__()
        self.classfication_block = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels)
        )
    
    def forward(self, x):
        x = self.classfication_block(x)

        return x

class FusionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionLayer, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):
        # x_mid_channels=256, x_global=256 size vector, fusion_feature_channels=256
        h = x1.shape[2]  # Height of a picture  
        w = x1.shape[3]  # Width of a picture

        x2 = torch.stack(tuple(x2 for _ in range(w)), 1)
        x2 = torch.stack(tuple(x2 for _ in range(h)), 1)
        x2 = x2.permute(0, 3, 1, 2)

        x_fusion = torch.cat((x1, x2), 1)
        # print(x_fusion.size())
        x_fusion = self.fusion_conv(x_fusion)
        # print(x_fusion.size())
        return x_fusion

class ColorizationNet(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, mode):
        super(ColorizationNet, self).__init__ ()
        self.color_block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU()
        )

        self.color_block2 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU()
        )

        self.color_block3 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels//4, in_channels//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//8),
            nn.Sigmoid()
        )

        self.output = nn.Conv2d(in_channels//8, out_channels, kernel_size=3, stride=1, padding=1)
        # self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        # print(x.size())
        x = self.color_block1(x)
        # print(x.size())
        x = self.color_block2(x)
        # print(x.size())
        x = self.color_block3(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())
        return x
    
class OutputLayer(nn.Module):
    def __init__(self, scale_factor, mode):
        super(OutputLayer, self).__init__()
        self.up =  nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x):
        x = self.up(x)
        # print(x_input.size())
        # x_out = torch.cat((x, x_input), 1)
        # print(x_out.size())

        return x

