# Facial-recognition-in-thermal-images
This repository contains my undergraduate research project, conducted at UFF, with the goal of learning how to use neural networks in thermal imaging.

The code was inspered in this site: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ 

## Pipeline
1. Image annotation
2. JSON to mask conversion
3. Dataset preparation
4. UNet training
5. Model saving
6. Prediction on new images
7. Visualization of results

## The Masks Converter
My masks were in JSON format, so I created a converter to get my masks in PNG. This converter is located in the file called "convert_masks.py"

## DatasetLoader
The SegmentationDataset do:
1. Image → tensor
2. Mask → tensor
3. Resize
4. Normalization

## The Training Flow
Dataset
   ↓
DataLoader
   ↓
U-Net
   ↓
Loss (BCEWithLogits)
   ↓
Backpropagation
   ↓
Atualização dos pesos
