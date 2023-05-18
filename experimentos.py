from maxMin import maxAndMin
from convNet import ConvNet
import cv2 as cv
import torch
import torch.nn as nn
import pyautogui
import os
import numpy as np
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dataLoad(path, want = 0):
    totalHolder = []
    dims = [1600,900]

    im = cv.cvtColor(cv.imread("eyes/300.450.27.jpg"), cv.COLOR_BGR2GRAY)
    top = max([max(x) for x in im])
    totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,torch.tensor([[int(("1400.150.7.jpg".split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder

trainingSet = dataLoad("eyes")


def eyetrack(xshift = 30, yshift=150, frameShrink = 0.15):
    model= ConvNet().to(device)
    model.load_state_dict(torch.load("xModels/108.plt",map_location=device))
    model.eval()


    im=trainingSet

    for i,(im, label) in enumerate(trainingSet):

        output=model(im)
        print(output.item())
        print(output.item()*1600)
 

    # top = max([max(x) for x in im])
    # im2=(torch.tensor([[im]]).to(dtype=torch.float,device=device))/top
    # lable=float(600/1600)

eyetrack()

   
    