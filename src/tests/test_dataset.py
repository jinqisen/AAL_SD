
import sys
import os
import torch
# 添加项目根目录到 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from core.dataset import Landslide4SenseDataset

def test_dataset():
    root_dir = Config.DATA_DIR
    
    print("Testing Train Set...")
    try:
        train_ds = Landslide4SenseDataset(root_dir, split='train')
        print(f"Train Dataset size: {len(train_ds)}")
        item = train_ds[0]
        img, mask = item["image"], item["mask"]
        print(f"Train Image Shape: {img.shape}, Dtype: {img.dtype}")
        print(f"Train Mask Shape: {mask.shape}, Dtype: {mask.dtype}")
        
        # 验证形状
        assert img.shape == (14, 128, 128), f"Expected image shape (14, 128, 128), got {img.shape}"
        assert mask.shape == (128, 128), f"Expected mask shape (128, 128), got {mask.shape}"
        assert isinstance(img, torch.Tensor), "Image should be a tensor"
        assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
        print("Train Set Check: PASSED")
    except Exception as e:
        print(f"Train Set Check: FAILED - {e}")

    print("\nTesting Valid Set...")
    try:
        val_ds = Landslide4SenseDataset(root_dir, split='val')
        print(f"Valid Dataset size: {len(val_ds)}")
        item = val_ds[0]
        img, mask = item["image"], item["mask"]
        assert img.shape == (14, 128, 128)
        assert mask.shape == (128, 128)
        print("Valid Set Check: PASSED")
    except Exception as e:
        print(f"Valid Set Check: FAILED - {e}")

    print("\nTesting Test Set...")
    try:
        test_img_dir = os.path.join(root_dir, "TestData", "img")
        if not os.path.isdir(test_img_dir):
            print(f"Test Set Check: SKIPPED - missing directory {test_img_dir}")
            return
        test_ds = Landslide4SenseDataset(root_dir, split='test')
        print(f"Test Dataset size: {len(test_ds)}")
        item = test_ds[0]
        img = item["image"]
        name = item["image_name"]
        mask = item["mask"]
        print(f"Test Image Name: {name}")
        assert img.shape == (14, 128, 128)
        assert isinstance(name, str)
        if int(mask.numel()) > 0:
            assert mask.shape == (128, 128)
        print("Test Set Check: PASSED")
    except Exception as e:
        print(f"Test Set Check: FAILED - {e}")

if __name__ == "__main__":
    test_dataset()
