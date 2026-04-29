from torch.utils.data import Dataset
import cv2 as cv
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        image = cv.imread(self.imagePaths[idx])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        mask = cv.imread(self.maskPaths[idx], 0)
        mask = (mask > 0).astype("float32")  # binária

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # garantir formato (1, H, W)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask