from maxMin import maxAndMin
from conv2Net import ConvNet
import face_recognition
import cv2 as cv
import torch
import torch.nn as nn
import pyautogui
import os
import numpy as np
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def process(im):
    eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    eye = cv.resize(eye, dsize=(100, 50))

    # Display the image - DEBUGGING ONLY
    #cv.imshow('frame', left_eye)

    top = max([max(x) for x in eye])
    eye = (torch.tensor([[eye]]).to(dtype=torch.float,
                                              device=device)) / top
    return eye

def dataLoad(path, want = 0):
    totalHolder = []
    dims = [1600,900]

    im = cv.cvtColor(cv.imread(path + "/" + "465.750.60.jpg"), cv.COLOR_BGR2GRAY)
    top = max([max(x) for x in im])
    totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,torch.tensor([[int(("465.621.60.jpg".split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder

#pruebaderecho = dataLoad("pruebaderecho")
#pruebaizquierdo = dataLoad("pruebaizquierdo")


def eyetrack(xshift = 30, yshift=150, frameShrink = 0.15):
    model= ConvNet().to(device)
    model.load_state_dict(torch.load("xModels/Sujetos_4_38_kernel_5.plt",map_location=device))
    model.eval()
    mvAvgx = []
    scale = 10
    margin = 100
    margin2 = 20

    webcam = cv.VideoCapture(0)
    while True:

        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)
        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
            reBds, leCenter = maxAndMin(feats[0]['right_eye'], mult=1/frameShrink)

            right_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye=frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]


            right_eye = process(right_eye)
            left_eye = process(left_eye)

            x=model(right_eye,left_eye)
            x=x.item()*1600
    

                
            pyautogui.moveTo(x,450)
 

    # top = max([max(x) for x in im])
    # im2=(torch.tensor([[im]]).to(dtype=torch.float,device=device))/top
    # lable=float(600/1600)

eyetrack()