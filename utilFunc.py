import csv
import gzip
import json

def readCSV(inpFile,delimiter=","):
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def readCompressedJson(inputFile):
    with gzip.open(inputFile, "rt", encoding="utf-8") as fIn:
        return json.load(fIn)