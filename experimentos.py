# from maxMin import maxAndMin
# from convNet import ConvNet
# import cv2 as cv
# import torch
# import torch.nn as nn
# import pyautogui
# import os
# import numpy as np
# import copy

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def dataLoad(path, want = 0):
#     totalHolder = []
#     dims = [1600,900]

#     im = cv.cvtColor(cv.imread("eyes/1500.50.17.jpg"), cv.COLOR_BGR2GRAY)
#     top = max([max(x) for x in im])
#     totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,torch.tensor([[int(("1400.150.7.jpg".split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

#     # print(totalHolder)
#     return totalHolder

# trainingSet = dataLoad("eyes")


# def eyetrack(xshift = 30, yshift=150, frameShrink = 0.15):
#     model= ConvNet().to(device)
#     model.load_state_dict(torch.load("xModels/118.plt",map_location=device))
#     model.eval()


#     im=trainingSet

#     for i,(im, label) in enumerate(trainingSet):

#         output=model(im)
#         print(output.item())
#         print(output.item()*1600)
 

#     # top = max([max(x) for x in im])
#     # im2=(torch.tensor([[im]]).to(dtype=torch.float,device=device))/top
#     # lable=float(600/1600)

# eyetrack()

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


num_epochs = 10
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
    
            # Reiniciar los gradientes
            if (i+1) % 1800 == 0:
                # testSc = evaluateModel(model,test,sidelen=900)
                testSc = evaluateModel(model,testderecho,testizquierdo)
                # trainSc = evaluateModel(model,trainingSet,sidelen=900)
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
    # finalScore = evaluateModel(bestModel,test,sidelen=900)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "xModels/" + str(int(finalScore))+".plt")



trainModel()




   
    