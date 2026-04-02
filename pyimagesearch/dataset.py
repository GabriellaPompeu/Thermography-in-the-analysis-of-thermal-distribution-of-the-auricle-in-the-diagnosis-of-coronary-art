from torch.utils.data import Dataset 
import cv2 as cv

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    # Return the number of total samples contained in the dataset
    def __len__(self):
        return len(self.imagePaths)
    
    # 
    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv.imread(imagePath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.maskPaths[idx], 0)

        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)
    
    
