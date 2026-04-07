# Importing the necessary packages
import torch 
import os

# Creating the path of the dataset
DATASET_PATH = os.path.join("dataset", "train")

# Defining the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks_png")

# Defining the fraction of the dataset I will keep aside for the test set
TEST_SPLIT = 0.15

# GPU or CPU?
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# Defining some important numbers
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# Initializing training parameters
INIT_LR = 0.0001
NUM_EPOCHS = 40
BATCH_SIZE = 2

# Defining the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# Defining threshold to filter weak predictions
THRESHOLD = 0.5

# Defining the path to the base output directory
BASE_OUTPUT = "output"

# Defining the path to the output serialized model, model training plot and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_path.txt"])
