# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import os

def prepare_plot(origImage, origMask, predMask):
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)

	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	# set the layout of the figure and display it
	figure.tight_layout()
	plt.show()

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv.imread(imagePath)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		image = cv.resize(image, (128, 128))
		orig = image.copy()

		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(
			config.MASK_DATASET_PATH,
			filename.replace(".jpg", ".png")
		)
		
		# loading the ground-truth segmentation mask in grayscale mode and resize it
		gtMask = cv.imread(groundTruthPath, 0)
		gtMask = cv.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
        
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		prepare_plot(orig, gtMask, predMask)

# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)