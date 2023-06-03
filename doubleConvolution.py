import torch.nn as nn

class doubleConvolution(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(doubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputChannels, outputChannels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace = True),
            nn.Conv2d(outputChannels, outputChannels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(outputChannels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        return self.conv(x)