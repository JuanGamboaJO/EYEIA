
import numpy as np
import cv2 as cv
import os
import copy
import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import matplotlib.pyplot as plt
from maxMin import maxAndMin
from conv2Net import ConvNet
from torch.optim.lr_scheduler import CyclicLR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dataLoad(path, want = 0):
    nameList = os.listdir(path)

    try:
        nameList.remove(".DS_Store")
    except:
        pass
    totalHolder = []
    dims = [1600,900]

    for name in nameList:
        im = cv.cvtColor(cv.imread(path + "/" + name), cv.COLOR_BGR2GRAY)
        top = max([max(x) for x in im])
        totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,
                            torch.tensor([[int((name.split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder

def evaluateModel(model,testSetderecho,testsetizquierdo, sidelen = 1600):
    model.eval()
    err = 0
    for i,((im, label),(im2,label2)) in enumerate(zip(testSetderecho,testsetizquierdo)):
        output = model(im,im2)
        err += abs(output.item() - label.item())
    model.train()

    return (err/len(testSetderecho)*sidelen)


trainingSet_Derecho = dataLoad("ojoderecho")
trainingSet_izquierdo = dataLoad("ojoizquierdo")
testizquierdo = dataLoad("testeyesizquierdo")
testderecho = dataLoad("testeyesderecho")


num_epochs = 20
bigTest = []
bigTrain = []

def trainModel():
    model = ConvNet().to(device)
    # model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='triangular')

    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        np.random.shuffle(trainingSet_Derecho)
        np.random.shuffle(trainingSet_izquierdo)

        for i,((im, label),(im2,label2)) in enumerate(zip(trainingSet_Derecho,trainingSet_izquierdo)):
            output = model(im,im2)
            #output = torch.mean(output, dim=1, keepdim=True)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    
            #  Ajustar la tasa de aprendizaje
            scheduler.step()
            optimizer.zero_grad()
    
            if (i+1) % 2520 == 0:
                #testSc = evaluateModel(model,testderecho,testizquierdo,sidelen=900)
                testSc = evaluateModel(model,testderecho,testizquierdo)
                #trainSc = evaluateModel(model,trainingSet_Derecho,trainingSet_izquierdo,sidelen=900)
                trainSc = evaluateModel(model,trainingSet_Derecho,trainingSet_izquierdo)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
          
                testscores.append(testSc)
                trainscores.append(trainSc)

                print(trainSc)
                print(testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(trainingSet_Derecho), loss.item()))

    bigTest.append(testscores)
    bigTrain.append(trainscores)

    finalScore = evaluateModel(bestModel,testderecho,testizquierdo)
    #finalScore = evaluateModel(bestModel,testderecho,testizquierdo,sidelen=900)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")



trainModel()




   
    