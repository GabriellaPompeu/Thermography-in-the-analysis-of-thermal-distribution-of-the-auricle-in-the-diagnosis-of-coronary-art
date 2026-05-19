from core import config
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
        return None, None, None, None

    return np.mean(region), np.min(region), np.max(region), np.std(region)

def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    filtered = np.zeros_like(mask)
    filtered[labels == largest_label] = 1

    return filtered

def save_results_table(ious, dices, accs, temps_mean, temps_min, temps_max):
    data = [
        ["IoU", f"{np.mean(ious):.4f}", f"{np.std(ious):.4f}"],
        ["Dice", f"{np.mean(dices):.4f}", f"{np.std(dices):.4f}"],
        ["Accuracy", f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}"],
    ]

    if len(temps_mean) > 0:
        data.extend([
            ["Temp Mean", f"{np.mean(temps_mean):.4f}", f"{np.std(temps_mean):.4f}"],
            ["Temp Min", f"{np.mean(temps_min):.4f}", f"{np.std(temps_min):.4f}"],
            ["Temp Max", f"{np.mean(temps_max):.4f}", f"{np.std(temps_max):.4f}"],
        ])

    columns = ["Métrica", "Média", "Desvio Padrão"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    table = ax.table(cellText=data, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)

    os.makedirs(config.BASE_PRED, exist_ok=True)
    plt.savefig(os.path.join(config.BASE_PRED, "results_table.png"), bbox_inches='tight')
    plt.close()

def plot_temperature_per_image(temps_mean, temps_min, temps_max, temps_std):
    import matplotlib.pyplot as plt

    x = range(len(temps_mean))

    plt.figure()

    # média com desvio padrão
    plt.errorbar(
        x,
        temps_mean,
        yerr=temps_std,
        fmt='o',
        capsize=5,
        label="Mean ± Std"
    )

    plt.plot(x, temps_min, linestyle='--', label="Min")
    plt.plot(x, temps_max, linestyle='--', label="Max")

    plt.fill_between(x, temps_min, temps_max, alpha=0.2)

    plt.xlabel("Image Index")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature per Image")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(config.BASE_PRED, "temperature_per_image.png"))
    plt.close()

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

def plot_prediction_quality(orig, gt, pred, iou, dice, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(orig)
    ax[0].set_title("Imagem")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title(f"Predição\nIoU: {iou:.3f} | Dice: {dice:.3f}")
    ax[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def make_predictions(model, imagePath):
	model.eval()
	with torch.no_grad():
		image = cv.imread(imagePath)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		orig = image.copy()  # imagem original
		(H, W) = image.shape[:2]  # tamanho original

		image = image.astype("float32") / 255.0
		image = cv.resize(image, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(
			config.MASK_DATASET_PATH,
			filename.replace(".jpg", ".png")
		)
		
		gtMask = cv.imread(groundTruthPath, 0)
		gtMask = cv.resize(gtMask, (W, H), interpolation=cv.INTER_NEAREST)
		
		gtMask_bin = (gtMask > 0).astype(np.uint8)

		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		predMask_bin = (predMask > config.THRESHOLD).astype(np.uint8)
		predMask_bin = keep_largest_component(predMask_bin)
		predMask_bin = cv.resize(predMask_bin, (W, H), interpolation=cv.INTER_NEAREST)
		predMask_vis = predMask_bin * 255

		os.makedirs(config.BASE_PRED, exist_ok=True)
		output_path = os.path.join(config.BASE_PRED, filename.replace(".jpg", ".png"))
		cv.imwrite(output_path, predMask_vis)

		overlay = orig.copy()
		contours, _ = cv.findContours(predMask_bin.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

		cv.drawContours(overlay, contours, -1, (0, 255, 0), 3)  # verde, espessura 2

		cv.imwrite(
			os.path.join(config.BASE_PRED, filename.replace(".jpg", "_overlay.png")),
			cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
		)

		thermal = load_thermal_data(imagePath)
		thermal = cv.resize(thermal, (W, H), interpolation=cv.INTER_NEAREST)

		iou, dice, acc = compute_metrics(predMask_bin, gtMask_bin)
		plot_prediction_quality(orig, gtMask, predMask_vis, iou, dice, save_path=os.path.join(config.BASE_PRED, filename.replace(".jpg", "_plot.png")))
		temp_mean, temp_min, temp_max, temp_std = extract_thermal_features_flir(thermal, predMask_bin)

		print("====Temperaturas em Celsius====\n")
		print(f"mean: {temp_mean:.2f} | min: {temp_min:.2f} | max: {temp_max:.2f}")
		print(f"IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {acc:.4f}")
		
		prepare_plot(orig, gtMask, predMask_vis)

		return iou, dice, acc, temp_mean, temp_min, temp_max, temp_std

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
print("Load up model...")
unet = torch.load(config.MODEL_PATH, weights_only=False).to(config.DEVICE)

ious, dices, accs = [], [], []
temps_mean, temps_min, temps_max, temps_std = [], [], [], []

for path in imagePaths:
	result = make_predictions(unet, path)

	if result is not None:
		iou, dice, acc, t_mean, t_min, t_max, t_std = result

		ious.append(iou)
		dices.append(dice)
		accs.append(acc)

		if t_mean is not None:
			temps_mean.append(t_mean)
			temps_min.append(t_min)
			temps_max.append(t_max)
			temps_std.append(t_std)

save_results_table(ious, dices, accs, temps_mean, temps_min, temps_max)
plot_temperature_per_image(temps_mean, temps_min, temps_max, temps_std)

print("\n=== RESULTADOS MEDIOS ===")
print(f"IoU medio: {np.mean(ious):.4f}")
print(f"Dice medio: {np.mean(dices):.4f}")
print(f"Acc media: {np.mean(accs):.4f}")
