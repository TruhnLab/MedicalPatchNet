import torchvision
import torch.nn as nn
import torch

class ScalePatchNet(nn.Module):
    def __init__(self,patchSize,outFeatures=512) -> None:
        super(ScalePatchNet, self).__init__()
        self.baseBackbone = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.baseBackbone.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.baseBackbone.classifier[1] = nn.Linear(in_features=1280, out_features=outFeatures, bias=True)
        self.patchSize = patchSize

    def forward(self,x): #normal forward to train 
        x = self.forwardRawPatches(x)
        x = torch.mean(x,dim=1)
        return x
    
    def forwardRawPatches(self,x): #forward raw patches
        batchSize = x.size()[0]
        #put patches in batch dimension
        x = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, self.patchSize, self.patchSize)
        x = self.baseBackbone(x)
        assert x.size()[0]%batchSize == 0
        #recreate image batches
        patchCount = int(x.size()[0]/batchSize)
        batchList = torch.split(x,patchCount)
        x = torch.stack(batchList)
        return x
    
    def forwardScaledPatches(self,x):
        rawPatchLogits = self.forwardRawPatches(x)
        globalLogits = torch.mean(rawPatchLogits,dim=1)
        globalProb = torch.nn.functional.sigmoid(globalLogits)
        globalProb = globalProb.unsqueeze(1).unsqueeze(2)
        scaledPatches = rawPatchLogits * globalProb
        return scaledPatches

    
MODEL_CLASS_LIST = [ #To make it easy to add other models
    ScalePatchNet,
]

def getModelClass(modelClassName):
    for ModelClass in MODEL_CLASS_LIST:
        if ModelClass.__name__ == modelClassName: return ModelClass
    assert False, modelClassName + "not in" + str([x.__name__ for x in MODEL_CLASS_LIST])
