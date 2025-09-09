
from torch.utils.data import Dataset
import torch
from functools import lru_cache

DEVICE = "cuda"

IMG_SIZE = 1024
PATCH_SIZE = 64

@lru_cache
def getShiftSteps(stepNum, patchSize):
    steps = [round(patchSize*(x/stepNum)) for x in range(stepNum)]
    steps = [((x + patchSize//2) % patchSize)-patchSize//2 for x in steps] # shift the upper half into the negative range
    steps.sort()
    return steps


def createShiftImg(img, shiftX, shiftY, patchSize, imgSize, useChannelDim=False):
    padAdd = patchSize//2
    
    shiftX = shiftX+padAdd
    shiftY = shiftY+padAdd

    padImg = torch.nn.functional.pad(img,(padAdd,padAdd,padAdd,padAdd),mode="constant",value=0)
    if useChannelDim:
        shiftImg = padImg[:,shiftX:shiftX+imgSize,shiftY:shiftY+imgSize]
    else:
        shiftImg = padImg[shiftX:shiftX+imgSize,shiftY:shiftY+imgSize]
    return shiftImg


class ShiftImgSet(Dataset):
    def __init__(self,img,shiftStepNum,patchSize=PATCH_SIZE,imgSize=IMG_SIZE):
        assert img.shape[0] == 1 # only one image in batch => batch dim can be deleted
        self.img = img[0]
        self.patchSize = patchSize
        self.imgSize = imgSize
        shiftSteps = getShiftSteps(shiftStepNum, patchSize)
        self.shiftTupList = list()
        for xShift in shiftSteps:
            for yShift in shiftSteps:
                self.shiftTupList.append((xShift,yShift))
    
    def __len__(self):
        return len(self.shiftTupList)
    
    def __getitem__(self, idx):
        shiftTup = self.shiftTupList[idx]
        shiftImg = createShiftImg(self.img, *shiftTup, patchSize=self.patchSize, imgSize=self.imgSize, useChannelDim=True)
        return shiftImg, torch.tensor(shiftTup)