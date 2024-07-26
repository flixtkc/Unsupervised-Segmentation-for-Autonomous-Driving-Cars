import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms

class CARLADataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = sorted(glob(os.path.join(root, 'imgs', split, '*.jpeg')))
        self.labels = sorted(glob(os.path.join(root, 'labels', split, '*.png')))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def __len__(self):
        return len(self.images)
