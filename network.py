import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def weights_init(m):
    """
    初始化权重：从 N(0, 0.02) 的正态分布采样
    适用于：Conv2d, ConvTranspose2d, BatchNorm2d
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.normal_(m.weight.data, 0.0, 0.02)  # 均值0，标准差0.02
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)  # 偏置初始化为0

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (3, 64, 64)
            nn.Conv2d(3, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # 添加Dropout（概率0.3）
            
            # (128, 32, 32)
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # (256, 16, 16)
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3), 

            # (512, 8, 8)
            nn.Conv2d(512, 1024, 5, 2, 2, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (1024, 4, 4)
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x).view(-1, 1)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 输入: (100, 1, 1)
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),  # 轻微Dropout（概率0.1）
            
            # (1024, 4, 4)
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # (512, 8, 8)
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # (256, 16, 16)
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # (128, 32, 32)
            nn.ConvTranspose2d(128, 3, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)