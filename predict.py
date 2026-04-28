from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import os
from flirimageextractor import FlirImageExtractor

def load_thermal_data(image_path):
    fie = FlirImageExtractor()
    fie.process_image(image_path)

    thermal = fie.get_thermal_np()
    return thermal

def extract_thermal_features_flir(thermal, mask):
    region = thermal[mask == 1]

    if len(region) == 0:
        return None, None, None

    return np.mean(region), np.min(region), np.max(region)

def prepare_plot(origImage, origMask, predMask):
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)

	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	figure.tight_layout()
	plt.show()

def make_predictions(model, imagePath):
	model.eval()
	with torch.no_grad():
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
		
		gtMask = cv.imread(groundTruthPath, 0)
		gtMask = cv.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))
		
		gtMask_bin = (gtMask > 0).astype(np.uint8)

		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		predMask_bin = (predMask > config.THRESHOLD).astype(np.uint8)
		predMask_vis = predMask_bin * 255

		os.makedirs(config.BASE_PRED, exist_ok=True)
		output_path = os.path.join(config.BASE_PRED, filename.replace(".jpg", ".png"))
		cv.imwrite(output_path, predMask_vis)

		overlay = orig.copy()
		overlay[predMask_bin == 1] = [255, 0, 0]  

		cv.imwrite(
			os.path.join(config.BASE_PRED, filename.replace(".jpg", "_overlay.png")),
			cv.cvtColor((overlay * 255).astype("uint8"), cv.COLOR_RGB2BGR)
		)

		thermal = load_thermal_data(imagePath)
		thermal = cv.resize(thermal, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))

		iou, dice, acc = compute_metrics(predMask_bin, gtMask_bin)
		temp_mean, temp_min, temp_max = extract_thermal_features_flir(thermal, predMask_bin)

		print("====Temperaturas em Celsius====\n")
		print(f"mean: {temp_mean:.2f} | min: {temp_min:.2f} | max: {temp_max:.2f}")
		print(f"IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {acc:.4f}")
		
		prepare_plot(orig, gtMask, predMask_vis)

		return iou, dice, acc

def compute_metrics(pred, gt):
    pred = pred > 0
    gt = gt > 0

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = intersection / (union + 1e-8)

    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)

    accuracy = (pred == gt).sum() / pred.size

    return iou, dice, accuracy

print("Loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
# imagePaths = np.random.choice(imagePaths, size=10)
print("Load up model...")
unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)

ious, dices, accs = [], [], []

for path in imagePaths:
	result = make_predictions(unet, path)
	if result is not None:
		iou, dice, acc = result
		ious.append(iou)
		dices.append(dice)
		accs.append(acc)

print("\n=== RESULTADOS MEDIOS ===")
print(f"IoU medio: {np.mean(ious):.4f}")
print(f"Dice medio: {np.mean(dices):.4f}")
print(f"Acc media: {np.mean(accs):.4f}")