from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d 
from torch.nn import MaxPool2d
from torch.nn import Module 
from torch.nn import ModuleList 
from torch.nn import ReLU
from torchvision.transforms import CenterCrop 
from torch.nn import functional as F 
import torch 

# Model Unet = Encoder class + Decoder class
# The Encoder will reduce the spatial dimension to compress information and will increase the number of channels
# The Decoder will take the final encoder representation, increase the spatial dimension and reduce the number of channels

class Block(Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, inChannels, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)]
        )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        blockOutputs = []

        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        
        return blockOutputs
    
class Decoder # Continue
