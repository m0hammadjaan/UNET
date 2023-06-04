import torch as T
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data.dataset as dataset
import os
from PIL import Image
import random

class TumorDataset(dataset):
    def __init__(self, rootDir, transform = True, debug = False):
        super(TumorDataset, self).__init__()
        self.rootDir = rootDir
        self.transform = {
            'hflip' : TF.hflip,
            'vflip' : TF.vflip,
            'rotate' : TF.rotate,
        }
        self.defaultTransform = transforms.compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])
        self.debug = debug
        if not transform:
            self.transform = None

    def getItem(self, index):
        imageName = os.path.join(self.rootDir, f'{index}.png')
        maskName = os.path.join(self.rootDir, f'{index}_mask.png')
        image = Image.open(imageName)
        mask = Image.open(maskName)
        if self.transform:
            image, mask = self.randomTransform(image, mask)

        image = TF.toTensor(image)
        mask = TF.toTensor(mask)

        sample = {
            'index' : int(index),
            'image' : image,
            'mask' : mask
        }
        return sample
    
    def randomTransform(self, image, mask):
        choiceList = list(self.transform.keys())
        for i in range(len(choiceList)):
            choiceKey = random.choice(choiceList)
            if self.debug:
                print(f'Applying {choiceKey} transformation')
            actionProbability = random.randint(0, 1)
            if actionProbability >= 0.5:
                if self.debug:
                    print(f'Applying {choiceKey} transformation')
                if choiceKey == 'rotate':
                    rotation = random.randint(15, 75)
                    if self.debug:
                        print(f'Rotating by {rotation} degrees')
                    image = self.transform[choiceKey](image, rotation)
                    mask = self.transform[choiceKey](mask, rotation)
                else:
                    image = self.transform[choiceKey](image)
                    mask = self.transform[choiceKey](mask)
            choiceList.remove(choiceKey)
        return image, mask
    
    def __len__(self):
        return len(os.listdir(self.rootDir)) // 2
    



