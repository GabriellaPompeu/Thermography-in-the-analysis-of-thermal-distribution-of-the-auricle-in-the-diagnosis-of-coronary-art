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
        return None, None, None, None

    return np.mean(region), np.min(region), np.max(region), np.std(region)

def save_results_table(ious, dices, accs, temps_mean, temps_min, temps_max):
    data = [
        ["IoU", np.mean(ious), np.std(ious)],
        ["Dice", np.mean(dices), np.std(dices)],
        ["Accuracy", np.mean(accs), np.std(accs)],
    ]

    if len(temps_mean) > 0:
        data.extend([
            ["Temp Mean", np.mean(temps_mean), np.std(temps_mean)],
            ["Temp Min", np.mean(temps_min), np.std(temps_min)],
            ["Temp Max", np.mean(temps_max), np.std(temps_max)],
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

    # min e max (opcional)
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

def make_predictions(model, imagePath):
	model.eval()
	with torch.no_grad():
		image = cv.imread(imagePath)
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

		orig = image.copy()  # imagem original (sem resize)
		(H, W) = image.shape[:2]  # tamanho original

		image = image.astype("float32") / 255.0
		image = cv.resize(image, (128, 128))

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
		predMask_bin = cv.resize(predMask_bin, (W, H), interpolation=cv.INTER_NEAREST)
		predMask_vis = predMask_bin * 255

		os.makedirs(config.BASE_PRED, exist_ok=True)
		output_path = os.path.join(config.BASE_PRED, filename.replace(".jpg", ".png"))
		cv.imwrite(output_path, predMask_vis)

		overlay = orig.copy()
		contours, _ = cv.findContours(predMask_bin.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

		cv.drawContours(overlay, contours, -1, (0, 255, 0), 3)  # vermelho, espessura 2

		cv.imwrite(
			os.path.join(config.BASE_PRED, filename.replace(".jpg", "_overlay.png")),
			cv.cvtColor(overlay, cv.COLOR_RGB2BGR)
		)

		thermal = load_thermal_data(imagePath)
		thermal = cv.resize(thermal, (W, H), interpolation=cv.INTER_NEAREST)

		iou, dice, acc = compute_metrics(predMask_bin, gtMask_bin)
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
# imagePaths = np.random.choice(imagePaths, size=10)
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