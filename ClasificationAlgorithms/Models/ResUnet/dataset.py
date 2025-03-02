import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class DFUTissueDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, file_names=None, mask_names=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Si se proporcionan los nombres de archivos, úsalos; de lo contrario, lista el directorio.
        if file_names is not None:
            self.images = file_names
        else:
            self.images = sorted(os.listdir(image_dir))

        if mask_dir:
            if mask_names is not None:
                self.masks = mask_names
            else:
                self.masks = sorted(os.listdir(mask_dir))
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.masks:
            # Se asume que el nombre de la imagen coincide con el de la máscara.
            mask_path = os.path.join(self.mask_dir, self.images[index])
            mask = np.array(Image.open(mask_path).convert("P"), dtype=np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # Máscara vacía si no hay máscara

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # Asegurarse de que ambas sean tensores
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)  # PyTorch usa `long` para segmentación

        return image, mask