import os
import cv2
import torch
from torch.utils.data import Dataset

class TongueDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")

        # ✅ FAST SIZE (CRITICAL)
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
