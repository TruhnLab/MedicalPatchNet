import csv
from PIL import Image
import os
import multiprocessing as mp

CROP_SIZE = 1024

OUT_FOLDER = "/path/to/output/" #replace with your output path



def readCSV(inpFile,delimiter=","):
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def cropAndScaleImg(img,squareSize):
    width,height = img.size
    cropLen = min(width,height)
    widthMargin = width - cropLen
    heightMargin = height - cropLen
    img = img.crop((widthMargin//2,heightMargin//2,widthMargin//2+cropLen,heightMargin//2+cropLen))
    img = img.resize((squareSize,squareSize),Image.LANCZOS)
    return img

def procImg(tup):
    inpPath,outPath,outFolder = tup
    if os.path.isfile(outPath):
        return
    os.makedirs(outFolder,exist_ok=True)
    print("DO:",outPath)
    outDirPath = "/".join(outPath.split("/")[:-1])
    print(outDirPath)
    inpImg = Image.open(inpPath)
    outImg = cropAndScaleImg(inpImg,CROP_SIZE)
    os.makedirs(outDirPath,exist_ok=True)
    outImg = outImg.convert("L")
    outImg.save(outPath,format='JPEG', subsampling=0, quality=100)

def genImgPairs(inpFile):
    head,*data = readCSV(inpFile)
    data = [x[0].replace("CheXpert-v1.0/","") for x in data] # only replace if "CheXpert-v1.0/" is present
    data = [(INP_FOLDER + x,OUT_FOLDER + x,OUT_FOLDER + "/".join(x.split("/")[:-1]) + "/") for x in data]
    return data

def procImgList(dataFile):
    imgList = genImgPairs(dataFile)
    print("found",len(imgList),"files")
    with mp.Pool(processes=12) as pool:
        pool.map(procImg,imgList)
    print("DONE",dataFile)


if __name__ == "__main__":
    INP_FOLDER = "/mnt/hdd1/datasets/CheXpert-v1.0/" #for converting the original CheXpert dataset
    TRAIN_FILE = INP_FOLDER + "train.csv"
    VALID_FILE = INP_FOLDER + "valid.csv"
    procImgList(TRAIN_FILE)
    procImgList(VALID_FILE)

    INP_FOLDER = "/home/user/datasets/chexlocalize/CheXpert/" #for converting the CheXpert test data which is contained in chexlocalize dataset
    TEST_FILE = INP_FOLDER + "test_labels.csv"
    procImgList(TEST_FILE)
    print("FULLY DONE")


