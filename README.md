# Thermography in the analysis of thermal distribution of the auricle in the diagnosis of coronary artery disease.
This repository contains my undergraduate research project, conducted at UFF, with the goal of learning how to use neural networks in thermal imaging.
The program implements a complete semantic segmentation pipeline for thermal images, whose objective is to automatically identify a region of interest (for example, the edge of the ear) and, from it, extract quantitative information such as average, minimum, and maximum temperature. The code for the U-Net application was inspered in this site: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/ 

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

## The Training
In general, this pipeline implements a complete supervised training flow for image segmentation, integrating data preparation, variability enhancement, architecture definition, optimization, and continuous evaluation, providing a solid foundation for future system improvements and extensions. At the end of the training period, a graph is generated showing the evolution of training and test losses over the epochs, allowing for a visual analysis of the model's behavior, such as identifying overfitting or underfitting.
