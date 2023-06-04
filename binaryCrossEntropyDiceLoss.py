import torch
import torch.nn as nn
import torch.nn.functional as F
from diceLoss import DiceLoss

class BinaryCrossEntropyDiceLoss(nn.Module):
    def __init__(self, device):
        super(BinaryCrossEntropyDiceLoss, self).__init__()
        self.diceLoss = DiceLoss().to(device)

    def forward(self, predicted, target):
        return F.binary_cross_entropy(predicted, target) + self.diceLoss(predicted, target)