
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Landslide4SenseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, with_mask: bool = True, bg_undersample_ratio: float = 1.0):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g., /path/to/Landslide4Sense)
            split (str): 'train', 'val', 'test'
            transform (callable, optional): 可选的转换函数
            with_mask (bool): 是否加载 mask
            bg_undersample_ratio (float): 背景下采样率 (1.0 = 不下采样，<1.0 则随机忽略该比例的背景像素)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.with_mask = bool(with_mask)
        self.bg_undersample_ratio = bg_undersample_ratio
        
        # 确定子目录
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'TrainData', 'img')
            self.mask_dir = os.path.join(root_dir, 'TrainData', 'mask')
        elif split == 'val':
            self.img_dir = os.path.join(root_dir, 'ValidData', 'img')
            self.mask_dir = os.path.join(root_dir, 'ValidData', 'mask')
        elif split == 'test':
            self.img_dir = os.path.join(root_dir, 'TestData', 'img')
            candidate_mask_dir = os.path.join(root_dir, 'TestData', 'mask')
            self.mask_dir = candidate_mask_dir if os.path.exists(candidate_mask_dir) else None
        else:
            raise ValueError(f"Unknown split: {split}")
            
        # 获取文件列表
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"Missing image directory for split={split}: {self.img_dir}")
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.h5')])
        if not self.images:
            raise RuntimeError(f"No image files found for split={split}: {self.img_dir}")
        
        self._mask_by_id = {}
        if (not self.with_mask) and split != "test":
            self.mask_dir = None
        if self.mask_dir is not None and self.with_mask:
            if not os.path.isdir(self.mask_dir):
                if split == "test":
                    self.mask_dir = None
                else:
                    raise RuntimeError(f"Missing mask directory for split={split}: {self.mask_dir}")
            if self.mask_dir is not None:
                mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(".h5")])
                for mf in mask_files:
                    mid = os.path.splitext(mf)[0]
                    self._mask_by_id[mid] = mf
                    # Handle L4S naming convention: mask_X -> image_X mapping
                    if mid.startswith("mask_"):
                        corresponding_img_id = "image_" + mid[5:]
                        self._mask_by_id[corresponding_img_id] = mf

                if split != "test":
                    missing = []
                    for img_name in self.images:
                        sid = os.path.splitext(img_name)[0]
                        if sid not in self._mask_by_id:
                            missing.append(sid)
                            if len(missing) >= 3:
                                break
                    if missing:
                        raise RuntimeError(
                            f"Missing masks for split={split} (example_ids={missing}) dir={self.mask_dir}"
                        )
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        sample_id = os.path.splitext(img_name)[0]
        
        # 读取图像
        with h5py.File(img_path, 'r') as f:
            image = f['img'][()] # Shape: (H, W, C) = (128, 128, 14)
            
        # 读取掩码 (如果有)
        mask = None
        if self.mask_dir is not None and self.with_mask:
            mask_name = self._mask_by_id.get(sample_id)
            if mask_name is None:
                if self.split != "test":
                    raise RuntimeError(f"Mask missing for sample_id={sample_id} split={self.split}")
            else:
                mask_path = os.path.join(self.mask_dir, mask_name)
                with h5py.File(mask_path, 'r') as f:
                    if "mask" not in f:
                        raise RuntimeError(
                            f"Missing dataset key 'mask' in {mask_path} (keys={list(f.keys())})"
                        )
                    mask = f['mask'][()] # Shape: (H, W) = (128, 128)
        
        # 转换为 float32
        image = image.astype(np.float32)

        if mask is not None:
            mask = mask.astype(np.int64)
            if self.split == 'train' and self.bg_undersample_ratio < 1.0:
                # Randomly ignore background pixels (set to 255)
                # mask == 0 is background
                bg_mask = (mask == 0)
                # Keep 'bg_undersample_ratio' of background, drop others
                rand_vals = np.random.rand(*mask.shape)
                drop_mask = bg_mask & (rand_vals > self.bg_undersample_ratio)
                mask[drop_mask] = 255
        
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
            
        if mask is None:
            mask = torch.empty((0,), dtype=torch.long)

        return {
            "image": image,
            "mask": mask,
            "image_name": img_name,
            "sample_id": sample_id,
            "split": self.split,
        }
