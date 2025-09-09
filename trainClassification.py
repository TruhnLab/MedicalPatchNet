import argParser as ARG
ARG.initArgs() #init arguments before loading other modules

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from ChexpertDataset import ClassificationChexpertDataset
from NetworkModel import getModelClass
from tqdm import tqdm
import torchmetrics.functional as tmf

import wandb

import multiprocessing as mp

def initWandB():

    wandb.login()
    runVar = wandb.init(
        project="MEDICAL_PATCH_NET",
        config={
            "learning_rate": ARG.LEARNING_RATE,
            "epoch_num":ARG.EPOCH_NUM,
            "batch_size":ARG.BATCH_SIZE,
            "patch_size":ARG.PATCH_SIZE,
            "wandb_name":ARG.WANDB_NAME
        },
        name=ARG.WANDB_NAME
        )


def log(name,val,printLog=True,commit=False):
    if isinstance(val,torch.Tensor): val = val.item()
    if printLog: print("LOG",name,val)
    if ARG.USE_WANDB: wandb.log({name:val},commit=commit)

NUM_PROC = 24 #number of processes used to load data
DEFAULT_MODEL_PATH = "savedModels/"

TEST = "test"

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        trainLoader: DataLoader,
        validLoaderList: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        gpuId: int,
        saveEvery: int = 1,
    ) -> None:
        
        self.gpuId = gpuId
        self.model = model.to(gpuId)
        self.trainLoader = trainLoader
        self.validLoaderList = validLoaderList
        self.optimizer = optimizer
        self.saveEvery = saveEvery
        self.scheduler = scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.ampScaler = GradScaler()

    def _runBatch(self, img, label):
        self.optimizer.zero_grad()

        with autocast(device_type='cuda'):
            output = self.model(img)
            loss = self.criterion(output,label)
        
        self.ampScaler.scale(loss).backward()
        self.ampScaler.step(self.optimizer)
        self.ampScaler.update()

        self.scheduler.step()
        
        log("learningRate",self.scheduler.get_last_lr()[0],printLog=False)
        log("loss",loss,commit=True,printLog=False)

    def _runEpoch(self, epoch):
        self.model.train()
        stepNum = len(self.trainLoader)
        print(f"Epoch: {epoch} | Batchsize: {ARG.BATCH_SIZE} | Steps: {stepNum}")
        
        for i, (img,label) in enumerate(tqdm(self.trainLoader)):
            img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)
            label = label.to(self.gpuId,non_blocking=True)
            self._runBatch(img,label)

        for el in self.validLoaderList:
            if el[2] == TEST and (epoch != ARG.EPOCH_NUM - 1): continue #if a test set is given, only apply it to the last 
            self.validate(*el)

    def _saveCheckpoint(self, epoch):
        saveName = DEFAULT_MODEL_PATH + ARG.MODEL_SAVE_PATH
        torch.save(self.model, saveName) # save entire model
        torch.save(self.model.state_dict(), saveName.replace(".pt","_weights.pt")) # save weights only

        print(f"Epoch {epoch} | Checkpoint saved at {saveName}")

    def train(self, maxEpochs: int):
        for epoch in range(maxEpochs):
            log("epoch",epoch)
            self._runEpoch(epoch)
            if epoch % self.saveEvery == 0:
                self._saveCheckpoint(epoch)
        if ARG.USE_WANDB: wandb.log({},commit=True)
    
    def calculateThresholds(self, thresholdLoader):
        #this function optimizes the threshold based on the validation data, NOT on the test data, to prevent data leakage 
        self.model.eval()
        outputList = list()
        labelList = list()
        for img,label in tqdm(thresholdLoader,desc="optThreshold"):
            img = img.to(self.gpuId, non_blocking=True, dtype=torch.float)
            with torch.no_grad():
                output = self.model(img).detach().cpu()
            outputList.append(nn.functional.sigmoid(output))
            labelList.append(label.cpu())

        allOutput = torch.cat(outputList).to(dtype=torch.float32)
        allLabel = torch.cat(labelList).to(dtype=torch.int32)
        labelNameList = thresholdLoader.dataset.getLabelNames()
        thrsDict = dict()

        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:, i]
            targets = allLabel[:, i]
            fpr, tpr, thresh = tmf.roc(preds, targets, task="binary")
            optIdx = torch.argmax(tpr - fpr)
            thrsDict[labelName] = thresh[optIdx].item()

        print(thrsDict)
        self.model.train()
        return thrsDict

    
    def validate(self,dataLoader,datasetName,validType,thresholdValidLoader=None):
        self.model.eval()
        outputList = list()
        labelList = list()
        for img,label in tqdm(dataLoader):
            img = img.to(self.gpuId,non_blocking=True,dtype=torch.float)

            output = self.model(img).detach().to("cpu",non_blocking=True)
            label = label.to("cpu",non_blocking=True,dtype=torch.int)
            # studentVect = self.studentModel(studentImg).detach().to("cpu",non_blocking=True)
            outputList.append(output)
            labelList.append(label)
        
        allOutput = torch.cat(outputList)
        allOutput = nn.functional.sigmoid(allOutput) # apply sigmoid to logits
        allLabel = torch.cat(labelList)

        labelNameList = dataLoader.dataset.getLabelNames()

        avgDict = dict()
        #calculate threshold based on validation data, not based on training data
        if thresholdValidLoader is not None: thrsDict = self.calculateThresholds(thresholdValidLoader)
        print("LEN:",len(labelNameList),labelNameList,allOutput.size(),allLabel.size())
        for i,labelName in enumerate(labelNameList):
            preds = allOutput[:,i]
            targets = allLabel[:,i]

            def execMetric(func,name=None):
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary")
                log(datasetName+"/"+labelName + "/" + metricName,metr)
                if metricName not in avgDict: avgDict[metricName] = list()
                avgDict[metricName].append(metr)
            
            def execThrsMetric(func,threshold,name=None,execWithoutThreshold=True):
                if execWithoutThreshold: execMetric(func,name)
                metricName = func.__name__ if name is None else name
                metr = func(preds,targets,task="binary",threshold=threshold)
                log(datasetName+"/"+labelName + "/thrs_" + metricName,metr)
            
            def execBootstrapMetric(func,name=None,threshold=None,bootstrapNum = 100000):
                argList = [(preds,targets,func,threshold) for _ in range(bootstrapNum)]


                with mp.Pool(ARG.PROC_NUM) as pool:
                    metrList = pool.map(execOneBootStrap,argList)
                
                metrList.sort()

                lowConfIdx = round(len(metrList)*0.025)
                highConfIdx = round(len(metrList)*0.975)
                medianIdx = round(len(metrList)*0.5)

                meanVal = sum(metrList)/len(metrList)
                metricName = func.__name__ if name is None else name
                thrsTag = "" if threshold is None else "thrs_"
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_low_" + metricName,metrList[lowConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_high_" + metricName,metrList[highConfIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_median_" + metricName,metrList[medianIdx])
                log(datasetName+"/"+labelName + f"/{thrsTag}bootstrap_mean_" + metricName,meanVal)
                




            execMetric(tmf.auroc)

            if thresholdValidLoader is not None:
                optThrs = thrsDict[labelName]
                log(datasetName+"/"+labelName+"/opt_thrs",optThrs)
                
                execThrsMetric(tmf.accuracy,threshold=optThrs)
                execThrsMetric(tmf.precision,threshold=optThrs)
                execThrsMetric(tmf.recall,threshold=optThrs,name="sensitivity")
                execThrsMetric(tmf.specificity,threshold=optThrs)
                execThrsMetric(tmf.f1_score,threshold=optThrs)
            
            if validType == TEST:
                execBootstrapMetric(tmf.auroc)
                execBootstrapBoth = lambda func,name=None: (execBootstrapMetric(func,name=name),execBootstrapMetric(func,name=name,threshold=optThrs))
                execBootstrapBoth(tmf.accuracy)
                execBootstrapBoth(tmf.precision)
                execBootstrapBoth(tmf.recall,name="sensitivity")
                execBootstrapBoth(tmf.specificity)
                execBootstrapBoth(tmf.f1_score)


        for key in avgDict.keys():
            valList = avgDict[key]
            avgVal = sum(valList)/len(valList)
            log(datasetName+"/avg/"+key,avgVal)

        self.model.train()


def execOneBootStrap(inp):
    preds,targets,func,threshold = inp

    assert len(preds) == len(targets)
    valCount = len(preds)

    sampled_indices = torch.multinomial(torch.ones(valCount), valCount, replacement=True)
    sampledPreds = preds[sampled_indices]
    sampledTargets = targets[sampled_indices]

    if threshold is None:
        metr = func(sampledPreds,sampledTargets,task="binary").item()
    else:
        metr = func(sampledPreds,sampledTargets,task="binary",threshold=threshold).item()
    
    return metr



def prepareDataloader(dataset: Dataset, batchSize: int, training: bool):
    return DataLoader(
        dataset,
        batch_size=batchSize,
        num_workers=NUM_PROC,
        pin_memory=True,
        shuffle=training,
        drop_last=training,
    )

def loadModelFromCheckpoint(weightPath):
    ModelClass = getModelClass(ARG.MODEL_CLASS)
    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)
    loadedWeights = torch.load(weightPath,weights_only=False)
    loadedWeights = {k.replace("_orig_mod.",""):v for k,v in loadedWeights.items()}
    model.load_state_dict(loadedWeights)
    model = torch.compile(model)
    return model

def loadTrainObjs():
    trainSet = ClassificationChexpertDataset("train",augementImg=True)


    ModelClass = getModelClass(ARG.MODEL_CLASS)
    model = ModelClass(patchSize=ARG.PATCH_SIZE,outFeatures=14)
    #compile model for speedup
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(),lr=ARG.LEARNING_RATE)
    return trainSet, model, optimizer

def loadValidationLoaderList():
    validLoaderList = list()

    chexpertValidationSet = ClassificationChexpertDataset("validate",False)
    chexpertValidLoader = prepareDataloader(chexpertValidationSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertValidLoader,"val_Chexpert", "val", chexpertValidLoader))


    ########
    # This section adding the test set was added after the hyperparameter optimization on the validation set.
    # We added it for the final runs to then directly get the test results.
    # During testing different achitectures and approaches this was not part of the code and the test dataset was not touched
    chexpertTestSet = ClassificationChexpertDataset(TEST,False)
    chexpertTestLoader = prepareDataloader(chexpertTestSet,ARG.BATCH_SIZE,training=False)

    validLoaderList.append((chexpertTestLoader,"test_Chexpert", TEST, chexpertValidLoader)) # the chexpertValidLoader is used to calculate the optimal threshold based on the validation set
    #######

    return validLoaderList




def trainAndEval(device, totalEpochs, saveEvery, batchSize):
    trainSet, model, optimizer = loadTrainObjs()
    trainLoader = prepareDataloader(trainSet, batchSize, training=True)
    #pretrainLoader = prepareDataloader(pretrainSet, batchSize, training=True) if pretrainSet is not None else None
    validLoaderList = loadValidationLoaderList()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,total_steps=ARG.EPOCH_NUM*len(trainLoader),max_lr=ARG.LEARNING_RATE,pct_start=0.05)

    trainer = Trainer(
        model=model,
        trainLoader=trainLoader,
        validLoaderList=validLoaderList,
        optimizer=optimizer,
        scheduler=scheduler,
        gpuId=device,
        saveEvery=saveEvery,
        #validsPerEpoch=ARG.VALIDS_PER_EPOCH,
    )
    trainer.train(totalEpochs)

def evalOnly(device,modelPath):
    validLoaderList = loadValidationLoaderList()
    model = loadModelFromCheckpoint(modelPath)
    trainer = Trainer(
        model=model,
        trainLoader=None,
        validLoaderList=None,
        optimizer=None,
        scheduler=None,
        gpuId=device,
        saveEvery=-1,
        #validsPerEpoch=ARG.VALIDS_PER_EPOCH,
    )
    for el in validLoaderList:
        trainer.validate(*el)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    if ARG.USE_WANDB: initWandB()

    device = 0  # shorthand for cuda:0
    if ARG.EVAL_ONLY:
        assert ARG.MODEL_LOAD_PATH is not None, "provide a path to model weights"
        evalOnly(
            device=device,
            modelPath=ARG.MODEL_LOAD_PATH,
        )
    else:
        assert ARG.MODEL_LOAD_PATH is None, "continue training not implemented yet."
        trainAndEval(
            device=device,
            totalEpochs=ARG.EPOCH_NUM,
            saveEvery=1,
            batchSize=ARG.BATCH_SIZE,
        )
