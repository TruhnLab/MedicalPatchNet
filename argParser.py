import argparse

### Arguments saved as global variable
LEARNING_RATE = None
BATCH_SIZE = None
EPOCH_NUM = None
USE_WANDB = None
PATCH_SIZE = None
WANDB_NAME = None
MODEL_SAVE_PATH = None
MODEL_LOAD_PATH = None
LR_REDUCTION_FACTOR = None
IMG_SIZE = None
ON_HPC = None
MODEL_CLASS = None
PROC_NUM = None
EVAL_ONLY = None

def initArgs():
    parser = argparse.ArgumentParser(description='Process command-line arguments.')

    parser.add_argument("-lr",type=float,dest="lr",help="Learning rate (default 1e-4)",default=1e-4)
    parser.add_argument("-ep",type=int,dest="ep",help="Total number of epochs (default 30)",default=30)
    parser.add_argument("-bs",type=int,dest="bs",help="Batch size (default 1)",default=1)
    parser.add_argument("-wb",action="store_true",dest="wb",help="Use weights and biases")
    parser.add_argument("-patchSize",type=int,dest="patchSize",help="Used patch size in vision encoder (default 64)",default=64)
    parser.add_argument("-wb_name",type=str,dest="wb_name",help="name used for weights and biases run",default="NOT DEFINED")
    parser.add_argument("-savePath",type=str,dest="savePath",help="The path where the visionEncoder is saved (default savedModels/latestEncoder.pt)",default="savedModels/latestEncoder.pt")
    parser.add_argument("-loadPath",type=str,dest="loadPath",help="A .pt file to load the vision encoder from it")
    parser.add_argument("-lrFactor",type=float,dest="lrFactor",help="The learning rate reduction factor defines the scaling of the highest and lowest learning rate. -lr is the highest (default 1000)",default=1000)
    parser.add_argument("-imgSize",type=int,dest="imgSize",help="The size of the total image used (default 512)",default=512)
    parser.add_argument("-hpc",action="store_true",dest="onHPC",help="Set this for code running on the hpc cluster")
    parser.add_argument("-model",type=str,dest="modelClass",help="The model class which is used (default: ScalePatchNet)",default="ScalePatchNet")
    parser.add_argument("-pr",type=int,dest="pr",help="The number of processes used for multiprocessing tasks",default=24)
    parser.add_argument("-evalOnly",action="store_true",dest="evalOnly",help="Set this to only evaluate the model")



    args = parser.parse_args()

    global LEARNING_RATE
    global EPOCH_NUM
    global BATCH_SIZE
    global USE_WANDB
    global PATCH_SIZE
    global WANDB_NAME
    global MODEL_SAVE_PATH
    global MODEL_LOAD_PATH
    global LR_REDUCTION_FACTOR
    global IMG_SIZE
    global ON_HPC
    global MODEL_CLASS
    global PROC_NUM
    global EVAL_ONLY

    LEARNING_RATE = args.lr
    EPOCH_NUM = args.ep
    BATCH_SIZE = args.bs
    USE_WANDB = args.wb
    PATCH_SIZE = args.patchSize
    WANDB_NAME = args.wb_name
    MODEL_SAVE_PATH = args.savePath
    MODEL_LOAD_PATH = args.loadPath
    LR_REDUCTION_FACTOR = args.lrFactor
    IMG_SIZE = args.imgSize
    ON_HPC = args.onHPC
    MODEL_CLASS = args.modelClass
    PROC_NUM = args.pr
    EVAL_ONLY = args.evalOnly