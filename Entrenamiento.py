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
from convNet import ConvNet
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

def evaluateModel(model,testSet, sidelen = 1600):
    model.eval()
    err = 0
    for (im, label) in testSet:
        output = model(im)
        err += abs(output.item() - label.item())
    model.train()

    return (err/len(testSet)*sidelen)


trainingSet = dataLoad("eyes")
test = dataLoad("testeyes")


num_epochs = 20
bigTest = []
bigTrain = []

def trainModel():
    model = ConvNet().to(device)
    # model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='triangular')

    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        np.random.shuffle(trainingSet)

        for i,(im, label) in enumerate(trainingSet):

            output = model(im)
            #output = torch.mean(output, dim=1, keepdim=True)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    
            #  Ajustar la tasa de aprendizaje
            scheduler.step()
            optimizer.zero_grad()
    
            # Reiniciar los gradientes
            if (i+1) % 2430 == 0:
                # testSc = evaluateModel(model,test,sidelen=900)
                testSc = evaluateModel(model,test)
                # trainSc = evaluateModel(model,trainingSet,sidelen=900)
                trainSc = evaluateModel(model,trainingSet)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
          
                testscores.append(testSc)
                trainscores.append(trainSc)

                print(trainSc)
                print(testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(trainingSet), loss.item()))

    bigTest.append(testscores)
    bigTrain.append(trainscores)

    finalScore = evaluateModel(bestModel,test)
    # finalScore = evaluateModel(bestModel,test,sidelen=900)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")



trainModel()



