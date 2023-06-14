import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from diceLoss import DiceLoss
import numpy as np
from datetime import datetime
from time import time
from tensorboardX import SummaryWriter


class TumorClassifier():
    def __init__(self, device, model):
        self.model = model
        self.device = device
        self.logPath = datetime.now().strftime("%I-%M-%S_%p_on_%B_%d,_%Y")
    
    def train(self, epochs, trainLoader, miniBatch = None, learningRate = 0.0001, saveBest = None, plotImage = None):
        self.tb_writer = SummaryWriter(log_dir=f'logs/{self.log_path}')
        history = {'trainLoss' : []}
        lastLoss = 1000
        self.optimizer = optim.Adam(self.model.parameters(), lr = learningRate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, facto = 0.85, patience = 2, verbose = True)
        for epoch in range(epochs):
            startTime = time()
            epochLoss = self.trainEpoch(trainLoader, miniBatch)
            history['trainLoss'].append(epochLoss)
            self.tb_writer.add_scalar('Train Loss', epochLoss, epoch)
            self.tb_writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.scheduler.step(epochLoss)
            if plotImage:
                self.model.eval()
                self.plotImage(plotImage)
                self.model.train()

            timeTaken = time() - startTime
            print(f'Epoch: {epoch+1:03d},  ', end='')
            print(f'Loss:{epochLoss:.7f},  ', end='')
            print(f'Time:{timeTaken:.2f}secs', end='')
            if saveBest != None and lastLoss > epochLoss:
                self.saveModel(saveBest)
                lastLoss = epochLoss
                print(f'\tSaved at loss: {epochLoss:.10f}')
            else:
                print()

        return history
    
    def test(self, testLoader, threshold = 0.5):
        self.model.eval()
        testDataIndexes = testLoader.sampler.indices[:]
        dataLen = len(testDataIndexes)
        batchSize = testLoader.batch_size
        if batchSize != 1:
            raise Exception('Batch size must be 1 for testing')
        testLoader = iter(testLoader)
        while len(testDataIndexes) != 0:
            data = testLoader.next()
            index = int(data['index'])
            if index in testDataIndexes:
                testDataIndexes.remove(index)
            else:
                continue
            image = data['image'].view(1, 1, 512, 512).to(self.device)
            mask = data['mask']
            maskPred = self.model(image).cpu()
            maskPred = maskPred.numpy()
            mask = np.resize(mask, (1, 512, 512))
            maskPred = np.resize(maskPred, (1, 512, 512))

            meanValueScore += self.DiceLoss.diceCoefficient(maskPred, mask)
            meanValueScore = meanValueScore / dataLen
            self.model.train()
            return meanValueScore


            


    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def loadModel(self, path):
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location = self.device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)