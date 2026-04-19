import os
from PIL import Image
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class GlassesDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=64, transform=None):
        '''
        Args:
            csv_path, img_dir: Paths to the CSV file and image directory.
            img_size: Desired size to resize images to (img_size x img_size).
            transform: Optional transformations to apply to the images.
        '''
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def _get_img_path(self, img_id):
        return os.path.join(self.img_dir, f"face-{img_id}.png")

    def __getitem__(self, idx):
        
        img_id = self.df['id'][idx]
        label = self.df['glasses'][idx]

        img_path = self._get_img_path(img_id)
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv loads images in BGR format, convert to RGB

        if self.transform:
            img = Image.fromarray(img) # convert to PIL Image for torchvision transforms
            img = self.transform(img)
        else:
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype('float32') / 255.0 # Normalize to [0,1]

            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2, 0, 1) # currently (H, W, C) -> rearrange to (C, H, W) for torch

        label = torch.tensor(label, dtype=torch.long)

        return img, label
