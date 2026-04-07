from torch.utils.data import Dataset 
import cv2 as cv

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, img_transforms, mask_transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.image_transform = img_transforms
        self.mask_transform = mask_transforms

    # Return the number of total samples contained in the dataset
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.maskPaths[idx], 0)
        mask = mask.astype("float32") / 255.0

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return (image, mask)
    
    
