from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt 
import torch
import time
import os

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice

imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = []

for imgPath in imagePaths:
    filename = os.path.splitext(os.path.basename(imgPath))[0]
    maskPath = os.path.join(config.MASK_DATASET_PATH, filename + ".png")
    maskPaths.append(maskPath)

print("Images:", len(imagePaths))
print("Masks:", len(maskPaths))

split = train_test_split(imagePaths, maskPaths,
	test_size=config.TEST_SPLIT, random_state=42)

(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

print("Saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# image_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(
#         (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
#         interpolation=transforms.InterpolationMode.BILINEAR
#     ),
#     transforms.ToTensor()
# ])

# mask_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(
#         (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
#         interpolation=transforms.InterpolationMode.NEAREST
#     ),
#     transforms.ToTensor()
# ])

train_transform = A.Compose([
    A.Resize(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # 🔥 IMPORTANTE
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH),
    
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),  # 🔥 IMPORTANTE
    ToTensorV2()
])

trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=train_transform)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=val_transform)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

trainLoader = DataLoader(trainDS, shuffle=True,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=0)
testLoader = DataLoader(testDS, shuffle=False,
    batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
    num_workers=0)

unet = UNet().to(config.DEVICE)
# lossFunc = BCEWithLogitsLoss()
bceLoss = BCEWithLogitsLoss(
    pos_weight=torch.tensor([3.0]).to(config.DEVICE)  # ⚖️ balanceamento
)

diceLoss = DiceLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

H = {"train_loss": [], "test_loss": []}

print("Training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	unet.train()
	totalTrainLoss = 0
	totalTestLoss = 0

	for (i, (x, y)) in enumerate(trainLoader):
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		pred = unet(x)
		# loss = lossFunc(pred, y) 
		loss = bceLoss(pred, y) + diceLoss(pred, y)
		opt.zero_grad()
		loss.backward()
		opt.step()
		totalTrainLoss += loss
	
	with torch.no_grad():
		unet.eval()

		for (x, y) in testLoader:
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			pred = unet(x)
			totalTestLoss += (bceLoss(pred, y) + diceLoss(pred, y))

	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps

	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))

endTime = time.time()
print("Total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
torch.save(unet, config.MODEL_PATH)