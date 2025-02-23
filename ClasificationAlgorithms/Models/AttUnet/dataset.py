import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class DFUTissueDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform =  transform
        self.images = os.listdir(image_dir)
        if mask_dir:
            self.masks = os.listdir(mask_dir)
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.masks:
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert("P"), dtype=np.float32)
            # mask[mask == 255.0] = 1.0
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Crear una máscara vacía (que no se use nunca)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask