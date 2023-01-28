import torch
import torchio as tio
import numpy as np

from typing import Tuple
from torch.utils.data import Dataset

class AutopetDataloaderTio(Dataset): 
    """
    Dataloader class to preprocess data for segmentation using torchio.

    Args:
        ct_images, pet_images, suv_images, labels (str): paths to all images

    Returns:
        image (torch.Tensor): stacked ct and pet images
        label (np.ndarray): ground truth of image
    """
    def __init__(self, ct_images: str, pet_images: str, suv_images: str, labels: str) -> None:
        self.ct_images = ct_images
        self.pet_images = pet_images
        self.suv_images = suv_images
        self.labels = labels

        assert len(ct_images) == len(pet_images) == len(labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, np.ndarray]:
        ct_transforms = tio.Compose([
            tio.transforms.CropOrPad((192, 192, 144)),
            tio.transforms.RescaleIntensity(out_min_max=(-1,1), in_min_max=(-1000,1000)),
        ])

        pet_transforms = tio.Compose([
            tio.transforms.CropOrPad((192, 192, 144)),
            tio.transforms.ZNormalization()        
        ])

        label_transforms = tio.Compose([
            tio.transforms.CropOrPad((192, 192, 144)),
        ])

        subject = tio.Subject(
            ct_image = ct_transforms(tio.ScalarImage(self.ct_images[idx])),
            pet_image = pet_transforms(tio.ScalarImage(self.pet_images[idx])),
            label = label_transforms(tio.LabelMap(self.labels[idx])),
        )

        image = np.stack([subject.ct_image, subject.pet_image], axis=-1)   
        image = torch.from_numpy(image.astype(np.float32))

        return image, np.array(subject.label)

    def __len__(self):
        return len(self.ct_images)

