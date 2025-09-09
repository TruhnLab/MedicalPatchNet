#This is just a hacky version of the evalClassification.py script to generateImages
import imgRetrivalUtil
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision.transforms import v2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from utilFunc import readCSV
import pickle
import os

from NetworkModel import ScalePatchNet

CHEXLOCALIZE_BASE_PATH = "/home/user/datasets/chexlocalize/" #REPLACE: replace with your path
CHEXPERT_PATH = CHEXLOCALIZE_BASE_PATH + "CheXpert/"
MAP_OUT_FOLDER = "/home/user/workspace/ChexLocalizeEval/CheXlocalizeImg/output/" #REPLACE: replace with your path

RAW_PATCH_NET = "RawPatchNet"
SCALED_PATCH_NET = "ScaledPatchNet"

PATCH_TYPE = SCALED_PATCH_NET

NORM_VAL = 200

PATIENT_LIST = None
CLASS = None



def encodeClassImg(model, device, shiftLoader):
    imgFeatrueList = list()
    shiftTupList = list()

    for shiftImgBatch,shiftTupBatch in shiftLoader:
        shiftImgBatch = shiftImgBatch.to(device)
        if PATCH_TYPE == RAW_PATCH_NET:
            imgFeatures = model.forwardRawPatches(shiftImgBatch)
        elif PATCH_TYPE == SCALED_PATCH_NET:
            imgFeatures = model.forwardScaledPatches(shiftImgBatch)
        else: assert False
        imgFeatures = imgFeatures.detach()
        imgFeatrueList.append(imgFeatures)
        shiftTupList.append(shiftTupBatch)
    allImgFeatures = torch.cat(imgFeatrueList)
    allShiftTup = torch.cat(shiftTupList)
    return allImgFeatures,allShiftTup

def genLocalClassMap(imgFeatures, shiftTup,imgSize,patchSize):
    assert imgSize % patchSize == 0
    patchCount = int(imgSize / patchSize)
    elementImgCount = imgFeatures.shape[0]
    featureNum = imgFeatures.shape[-1]
    imgFeatures = torch.reshape(imgFeatures,(-1,patchCount,patchCount,featureNum))
    
    shiftTup = shiftTup * -1 # reverse shift, because relative to the image, the grid is shifted into the other direction
    
    totalMap = torch.zeros((featureNum,imgSize,imgSize),dtype=imgFeatures.dtype,device=imgFeatures.device)

    for i in range(elementImgCount):
        elementImg = imgFeatures[i]
        elementImg = torch.permute(elementImg,(2,0,1))# channel = features => first dimension
        elementShift = shiftTup[i].tolist()

        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=1)
        elementImg = torch.repeat_interleave(elementImg, patchSize, dim=2)
        elementImg = imgRetrivalUtil.createShiftImg(elementImg,*elementShift,patchSize=patchSize,imgSize=imgSize,useChannelDim=True)
        totalMap += elementImg
    totalMap = totalMap/elementImgCount
    return totalMap

def applyGradCamOneClass(model,img,targetId,CamAlg):
    target_layers = [model.baseBackbone.features[7]] # check if correct
    targets = [ClassifierOutputTarget(targetId)]
    with CamAlg(model=model,target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = torch.tensor(grayscale_cam)
        return grayscale_cam

def applyGradCam(model,img,targetCount,CamAlg):
    mapList = list()
    for i in range(targetCount):
        oneClassMap = applyGradCamOneClass(model,img,i,CamAlg=CamAlg)
        mapList.append(oneClassMap)
    allClassMap = torch.cat(mapList)
    return allClassMap, None

def applyPatchLocalisation(model,img, imgSize, patchSize ,stepsPerPatch, device,):
    shiftImgDataset = imgRetrivalUtil.ShiftImgSet(img,stepsPerPatch,patchSize,imgSize)
    shiftLoader = iter(DataLoader(shiftImgDataset,batch_size=4,shuffle=False))
    imgFeatures, shiftTup = encodeClassImg(model,device,shiftLoader)
    localClassMap = genLocalClassMap(imgFeatures,shiftTup,imgSize,patchSize)
    return localClassMap,(imgFeatures,shiftTup)

def getChexpertImgPath(line):
    path =  CHEXPERT_PATH + line[0]
    path = path.replace("CheXpert-v1.0/valid","val")
    return path

def scaleImg(img,size):
    return v2.Resize(size=size,antialias=True)(img)

def cropImg(img):
    width,height = img.size
    cropLen = min(width,height)
    widthMargin = width - cropLen
    heightMargin = height - cropLen
    img = img.crop((widthMargin//2,heightMargin//2,widthMargin//2+cropLen,heightMargin//2+cropLen))
    return img
    

def getImg(imgPath,crop=True):
    img = Image.open(imgPath).convert("L")
    origSize = img.size
    if crop: img = cropImg(img)
    img = torchvision.transforms.functional.to_tensor(img)
    return img,origSize

def fetchImg(line,size):
    imgPath = getChexpertImgPath(line)
    img,origSize = getImg(imgPath)

    scaledImg = scaleImg(img,size)
    return scaledImg,origSize


def genChexlocalizeMap(imgLine,head,model,mapFunc,outPath,split,imgSize,patch_size,device,stepsPerPatch=None,CamAlg=None,convertToPng=False,delPkl=False,addName=""):
    img,origSize = fetchImg(imgLine,(imgSize,imgSize))

    img = img.to(device,non_blocking=True)
    #label = label[None]
    img = img[None] # add batch dimension [batch size = 1]

    cutOff = 1 if split == "test" else 5
    #cutOff = 5
    labelNames = head[cutOff:]
    gtList = imgLine[cutOff:]
    allClassMap = None
    if mapFunc == applyGradCam:
        allClassMap,_ = applyGradCam(model,img,len(labelNames),CamAlg=CamAlg)
        allClassMap.detach().cpu()
    elif mapFunc == applyPatchLocalisation:
        allClassMap,_ = applyPatchLocalisation(model,img,imgSize=imgSize,patchSize=patch_size,stepsPerPatch=stepsPerPatch,device=device)
        #print(torch.min(allClassMap),torch.max(allClassMap),torch.mean(allClassMap))
        #allClassMap = torch.nn.functional.sigmoid(allClassMap)
        allClassMap = (torch.clip(allClassMap,-1*NORM_VAL,NORM_VAL) + NORM_VAL)/(2*NORM_VAL)
        print("range:",torch.min(allClassMap),torch.max(allClassMap))
        allClassMap = allClassMap.detach().cpu()
    assert allClassMap is not None

    
    modelGlobalOutput = model(img)[0] # batch size = 1
    #modelGlobalOutput = torch.nn.functional.sigmoid(modelGlobalOutput)
    modelGlobalOutput = modelGlobalOutput.detach().cpu()

    
    
    expandedCxr = img[0].expand(3,-1,-1).detach().cpu()

    

    for i,name in enumerate(labelNames):
        if CLASS is not None:
            if name != CLASS: continue
        if name == "Lung Opacity": name = "Airspace Opacity" # different label from Chexlocalize and Chexpert/MIMIC ( https://www.nature.com/articles/s42256-022-00536-x )
        groundTruth = int(float(gtList[i]))

        prob = torch.nn.functional.sigmoid(modelGlobalOutput[i]).item()
        
        partClassMap = allClassMap[i].detach().cpu()
        partClassMap = partClassMap[None][None]
        #if mapFunc == applyPatchLocalisation: partClassMap = partClassMap*prob => old => later scaling
        retDict = {
                'map': partClassMap,
                'prob': prob,
                'task': name,
                'gt':groundTruth,
                'cxr_img':expandedCxr,
                'cxr_dims': origSize,
            }
        
        fileDescList = imgLine[0].split("/")
        patient = fileDescList[-3]
        study = fileDescList[-2]
        view = fileDescList[-1].replace(".jpg","")
        pklFileName = outPath +  patient + "_" + study + "_" + view + "_" + name + "_map_"+str(stepsPerPatch)+".pkl"
        with open(pklFileName, 'wb') as handle:
            pickle.dump(retDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved",pklFileName)
        if convertToPng:
            pngFileName = pklFileName.replace(".pkl","_"+addName+".png")
            pklToPNG(pklFileName,pngPath=pngFileName)
        if delPkl: os.remove(pklFileName)


def pklToPNG(pklPath,pngPath=None):
    if pngPath is None:
        pngPath = pklPath.replace(".pkl",".png")
    isPatchNet = "PatchNet" in pklPath

    with open(pklPath, "rb") as f:
        data = pickle.load(f)

    map_val = data["map"][0][0]
    img = data["cxr_img"][0]
    prob = data["prob"]
    map_val = map_val.cpu().detach().numpy()
    if isPatchNet:
        map_val = map_val - 0.5 #in map the values are saved from 0 to 1 => set 0.5 to the new 0 => 
    
    if isPatchNet:
        arg = {"vmin":-0.2,"vmax":0.2} # scaling for visualization (it was already scaled by NORM_VAL)
    else:
        arg = {"vmin":-1.0,"vmax":1.0} #others are from 0 to 1 => no blue values which are from -1.0 to 0.0 as these methods dont have 
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.imshow(map_val*(-1), cmap='RdBu', alpha=0.5,**arg)# *(-1) => invert colormap
    plt.axis('off')
    #plt.title(file_name)
    plt.savefig(pngPath,bbox_inches='tight')
    print("created",pngPath)
    plt.close()

def execEval(modelPath,usedSplit,CamAlg=None,imgSize=None,patchSize=None,stepsPerPatch=16,device="cuda"):
    

    modelName = modelPath.split("/")[-1].replace(".pt","")



    assert usedSplit in ["test","val",None]
    
    
    # Load the weights from checkpoint and handle compiled model prefixes
    state_dict = torch.load(modelPath)
    state_dict = {k.replace("_orig_mod.",""):v for k,v in state_dict.items()} # the model was saved from a compiled model => prefix _orig_mod. for each key

    model = ScalePatchNet(patchSize=patchSize, outFeatures=14)
    model.load_state_dict(state_dict)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    head,*dataLines = readCSV(CHEXPERT_PATH + usedSplit + "_labels.csv")

    camName = PATCH_TYPE if CamAlg is None else CamAlg.__name__

    outFolder = MAP_OUT_FOLDER +usedSplit+"_" + camName + "_" + modelName + "/"
    os.makedirs(outFolder,exist_ok=True)

    mapFunc = applyPatchLocalisation if CamAlg is None else applyGradCam

    for line in tqdm(dataLines):
        found = False
        for p in PATIENT_LIST:
            if p in line[0]:
                print("found",found)
                found = True
        if not found: continue
        print(line)


        genChexlocalizeMap(line,head,model,mapFunc,outFolder,split=usedSplit,imgSize=imgSize,patch_size=patchSize,device=device,stepsPerPatch=stepsPerPatch,CamAlg=CamAlg,convertToPng=True,delPkl=True,addName=camName)




if __name__ == "__main__":

    
    PATIENT_LIST = [""] #generate images for all patients
    CLASS = None # All classes
    PATCH_TYPE = RAW_PATCH_NET # its a bit ugly to use a global variable as argument => TODO: change
    execEval("savedModels/MedicalPatchNet_weights.pt","val",stepsPerPatch=64,imgSize=512,patchSize=64)
    PATCH_TYPE = SCALED_PATCH_NET
    execEval("savedModels/MedicalPatchNet.pt","val",stepsPerPatch=64,imgSize=512,patchSize=64)

    execEval("savedModels/EfficientNetB0.pt","val",CamAlg=GradCAM,imgSize=512,patchSize=512)
    execEval("savedModels/EfficientNetB0.pt","val",CamAlg=GradCAMPlusPlus,imgSize=512,patchSize=512)
    execEval("savedModels/EfficientNetB0.pt","val",CamAlg=EigenCAM,imgSize=512,patchSize=512)
    
    print("DONE")