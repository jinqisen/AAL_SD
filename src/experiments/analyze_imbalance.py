import os
import sys
import h5py
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from core.dataset import Landslide4SenseDataset
from config import Config

cfg = Config()
ds = Landslide4SenseDataset(cfg.DATA_DIR, split='train')

pos_pixels = 0
total_pixels = 0
pos_images = 0
total_images = len(ds.images)

image_stats = []

for i, img_name in enumerate(ds.images):
    sample_id = os.path.splitext(img_name)[0]
    mask_name = ds._mask_by_id.get(sample_id)
    if mask_name is None:
        continue
    mask_path = os.path.join(ds.mask_dir, mask_name)
    with h5py.File(mask_path, 'r') as f:
        mask = np.array(f['mask'])
        pos = np.sum(mask > 0)
        tot = mask.size
        pos_pixels += pos
        total_pixels += tot
        if pos > 0:
            pos_images += 1
        image_stats.append((img_name, pos, tot))
    if i % 500 == 0:
        print(f'Processed {i}/{total_images}')

print(f'Total images: {total_images}')
print(f'Images with landslide: {pos_images}')
print(f'Total pixels: {total_pixels}')
print(f'Landslide pixels: {pos_pixels}')
print(f'Background pixels: {total_pixels - pos_pixels}')
print(f'Imbalance ratio (bg/fg): {(total_pixels - pos_pixels) / max(1, pos_pixels):.2f}')

with open('image_stats.pkl', 'wb') as f:
    pickle.dump(image_stats, f)
