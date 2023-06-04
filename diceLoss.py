import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, predicted, target):
        batch = predicted.size()[0]
        batchLoss = 0
        for i in range(batch):
            coefficient = self.diceCoefficient(predicted[i], target[i])
            batchLoss += coefficient
        batchLoss = batchLoss / batch
        return 1 - batchLoss
    
    def diceCoefficient(self, predicted, target):
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2 * intersection + smooth) / (predicted.sum() + target.sum() + smooth)
        return coefficient
