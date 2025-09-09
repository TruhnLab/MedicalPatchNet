from torch.utils.data import Dataset
from utilFunc import readCSV
import argParser as ARG
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms.functional
import torch

TEST = "test"
TRAIN = "train"
VALIDATE = "validate"

POSSIBLE_SPLITS = [
    TEST,
    TRAIN,
    VALIDATE,
]

if ARG.ON_HPC:
    BASE_PATH = "/hpcwork/p0021834/workspace_patrick/datasets/CheXpert-v1.0_1024/"
else:
    BASE_PATH = "/home/patrick/datasets/CheXpert-v1.0_1024/"

META_DICT = {
    TRAIN:"train_visualCheXbert.csv",
    #TRAIN:"train.csv", #for the classic training data
    VALIDATE:"valid.csv",
    TEST:"test_labels.csv",
}

META_LABEL_CUT = 5
META_LABEL_CUT_DICT = {
    TRAIN:5,
    VALIDATE:5,
    TEST:1,
}

def shiftVisualChexpert(csvCont): # align visualCheXpert label with the normal chexpert training label order
    labelCut = 5
    for line in csvCont:
        lastEl = line.pop()
        line.insert(labelCut,lastEl)
    print("SHIFTED VISUAL CHEXPERT TRAINING FILE")


class ClassificationChexpertDataset(Dataset):
    def __init__(self, usedSplit, augementImg=False, imgSize=ARG.IMG_SIZE):
        self.usedSplit = usedSplit
        self.augementImg = augementImg
        self.imgSize = imgSize
        metaFile = BASE_PATH + META_DICT[self.usedSplit]
        #head,*data = readCSV(metaFile)
        csvCont = readCSV(metaFile)
        if META_DICT[self.usedSplit] == "train_visualCheXbert.csv": shiftVisualChexpert(csvCont) #shift dataset
        head,*data = csvCont

        labelCut = META_LABEL_CUT_DICT[usedSplit]
        self.labelNames = head[labelCut:]

        getTotalPath = lambda relativePath: BASE_PATH + relativePath.replace("CheXpert-v1.0/","")

        labelDict = { #set unknown to 0 => there is no unknow in in the visual chexpert label
            "": 0,
            "1.0": 1,
            "0.0": 0,
            "-1.0": 0,
        }
        #labelToInt = lambda lst: [int(float(x)) for x in lst]
        labelToInt = lambda lst: [labelDict[x] for x in lst]
        self.itemList = [(getTotalPath(x[0]),labelToInt(x[labelCut:])) for x in data]
        self.augmentTransforms = v2.Compose([
            v2.RandomResizedCrop(size=(self.imgSize,self.imgSize),scale=(0.5,1),antialias=True),
            v2.RandomRotation(degrees=5),
            v2.ColorJitter(brightness=0.3),
        ])
        self.normalTransform = v2.Compose([
            v2.Resize(size=(self.imgSize,self.imgSize),antialias=True)
        ])

    def applyAugment(self,img):
        if not self.augementImg:
            return self.normalTransform(img)
        img = self.augmentTransforms(img)
        return img
    
    def getLabelNames(self):
        return self.labelNames
    
    def __len__(self):
        return len(self.itemList)
    
    def __getitem__(self, idx):
        imgPath,label = self.itemList[idx]
        img = self.getImg(imgPath)
        img = self.applyAugment(img)

        label = torch.tensor(label,dtype=torch.float32)

        return img,label

    def getImg(self,imgPath):
        img = Image.open(imgPath).convert("L")
        img = torchvision.transforms.functional.to_tensor(img)
        return img