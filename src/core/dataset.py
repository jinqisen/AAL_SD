
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Landslide4SenseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g., /path/to/Landslide4Sense)
            split (str): 'train', 'val', 'test'
            transform (callable, optional): 可选的转换函数
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 确定子目录
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'TrainData', 'img')
            self.mask_dir = os.path.join(root_dir, 'TrainData', 'mask')
        elif split == 'val':
            self.img_dir = os.path.join(root_dir, 'ValidData', 'img')
            self.mask_dir = os.path.join(root_dir, 'ValidData', 'mask')
        elif split == 'test':
            self.img_dir = os.path.join(root_dir, 'TestData', 'img')
            self.mask_dir = None
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # 获取文件列表
        if os.path.exists(self.img_dir):
            self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.h5')])
        else:
            self.images = []
        
        # 简单检查
        if self.mask_dir and os.path.exists(self.mask_dir):
            self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.h5')])
            if len(self.images) != len(self.masks):
                print(f"Warning: Images and masks count mismatch in {split} set: {len(self.images)} vs {len(self.masks)}")
        else:
            self.masks = []
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 读取图像
        with h5py.File(img_path, 'r') as f:
            image = f['img'][()] # Shape: (H, W, C) = (128, 128, 14)
            
        # 读取掩码 (如果有)
        mask = None
        if self.split != 'test' and self.masks:
            mask_name = self.masks[idx]
            mask_path = os.path.join(self.mask_dir, mask_name)
            with h5py.File(mask_path, 'r') as f:
                mask = f['mask'][()] # Shape: (H, W) = (128, 128)
        
        # 转换为 float32
        image = image.astype(np.float32)
        
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']

        # 如果没有 transform 或者 transform 返回的是 numpy，我们手动转换为 Tensor
        if not isinstance(image, torch.Tensor):
            # HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1)
            
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
            
        if self.split == 'test':
            return image, img_name
        
        if mask is None:
             # Fallback for unlabeled pool if mask missing
            return image
            
        return image, mask
