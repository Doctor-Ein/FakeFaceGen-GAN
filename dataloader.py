import os
import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader

img_dim = 128
PATH = "img/processed/"

class DataGenerater(Dataset):
    def __init__(self, path=PATH, transform=None):
        """
        修改点：
        1. 使用 os.path.join 安全拼接路径
        2. 添加扩展名过滤（仅支持 .png/.jpg/.jpeg）
        """
        self.dir = path.rstrip('/')  # 移除末尾可能存在的斜杠
        # 仅保留图片文件（不区分大小写）
        self.datalist = [
            f for f in os.listdir(self.dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.image_size = (img_dim, img_dim)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.datalist[idx]) # 安全路径拼接
        img = io.imread(img_path)
        img = transform.resize(img, self.image_size)
        img = img.transpose((2, 0, 1))  # 转换为通道优先
        img = img.astype("float32")

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.datalist)

train_dataset = DataGenerater()
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)