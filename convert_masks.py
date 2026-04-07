import os
import json
import numpy as np
import cv2
from glob import glob

json_dir = "dataset/train/masks"
output_dir = "dataset/train/masks_png"

os.makedirs(output_dir, exist_ok=True)

json_files = glob(os.path.join(json_dir, "*.json"))

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]

    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)

    filename = os.path.basename(json_file).replace(".json", ".png")
    output_path = os.path.join(output_dir, filename)

    cv2.imwrite(output_path, mask * 255)

print("Conversão concluída!")