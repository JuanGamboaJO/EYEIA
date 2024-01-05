import os
import cv2 as cv
import face_recognition
from maxMin import maxAndMin
import pyautogui
import copy
import random

pyautogui.FAILSAFE=False
def getEye(times = 1,frameShrink = 0.15, coords = (0,0), counterStart = 0, folder = "ojo"):
    webcam = cv.VideoCapture(0)
    counter = counterStart
    ims = []

    while counter < counterStart+times:
        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats = face_recognition.face_landmarks(smallframe)
        if len(feats) > 0:
            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
            reBds, leCenter = maxAndMin(feats[0]['right_eye'], mult=1/frameShrink)

            right_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye=frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]
            # right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

            right_eye = cv.cvtColor(right_eye, cv.COLOR_BGR2GRAY)
            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)

            right_eye = cv.resize(right_eye, dsize=(100, 50))
            left_eye = cv.resize(left_eye, dsize=(100, 50))

            # D
            # isplay the image - DEBUGGING ONLY
            #cv.imshow('frame', left_eye)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            cv.imwrite(folder +'izquierdo' + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(counter) + ".jpg", left_eye)
            print("Se guardo imagen del ojo izquierdo")
            cv.imwrite(folder +'derecho' + "/" + str(coords[0]) + "." + str(coords[1]) + "." + str(counter) + ".jpg", right_eye)
            print("Se guardo imagen del ojo derecho")
            counter += 1

def primeratoma(tomas,folder="ojo"):
    coords_list = [(0, 0), (800, 0), (1600, 0), (0, 450),(800, 450),(1600, 450),(0, 900),(800, 900),(1600, 900)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def segundatoma(tomas,folder="ojo"):
    coords_list = [(100, 50),(800, 50),(1500, 50),(100, 450),(1500, 450),(100, 850),(800, 850),(1500, 850)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def terceratoma(tomas,folder="ojo"):
    coords_list = [(200, 100),(800, 100),(1400, 100),(200, 450),(1400, 450),(200, 800),(800, 800),(1400, 800)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def cuartatoma(tomas,folder="ojo"):
    coords_list = [(300, 150),(800, 150),(1300, 150),(300, 450),(1300, 450),(300, 750),(800, 750),(1300, 750)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def quintatoma(tomas,folder="ojo"):
    coords_list = [(400, 200),(800, 200),(1200, 200),(400, 450),(1200, 450),(400, 700),(800, 700),(1200, 700)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def sextatoma(tomas,folder="ojo"):
    coords_list = [(500, 250),(800, 250),(1100, 250),(500, 450),(1100, 450),(500, 650),(800, 650),(1100, 650)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def septimatoma(tomas,folder="ojo"):
    coords_list = [(100,300 ),(250, 300),(400, 300),(100,600),(250, 600),(400, 600)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def octavatoma(tomas,folder="ojo"):
    coords_list = [(1500,300 ),(1350, 300),(1200, 300),(1500,600),(1350, 600),(1200, 600)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def novenatoma(tomas,folder="ojo"):
    coords_list = [(600,200 ),(600, 125),(600, 50),(600,700),(600, 775),(600, 850)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def decimatoma(tomas,folder="ojo"):
    coords_list = [(1000,200 ),(1000, 125),(1000, 50),(1000,700),(1000, 775),(1000, 850)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)

def onceavatoma(tomas,folder="ojo"):
    coords_list = [(600, 300),(600, 600),(1000, 600),(1000, 300),(600, 450),(1000, 450),(800, 300),(800, 600)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=20, coords=(i, j), counterStart=tomas, folder=folder)
    

def pruebatoma(folder="eyes"):
    coords_list = [(465, 750)]
    for i, j in coords_list:
        print(i, j)
        pyautogui.moveTo(i, j)
        input("Press Enter to continue...")
        pyautogui.moveTo(i, j)
        getEye(times=1, coords=(i, j), counterStart=60, folder="prueba")

contador=60
#primeratoma(tomas=contador)
#segundatoma(tomas=contador)
#terceratoma(tomas=contador)
#cuartatoma(tomas=contador)
#quintatoma(tomas=contador)
#sextatoma(tomas=contador)
#septimatoma(tomas=contador)
#octavatoma(tomas=contador)
#novenatoma(tomas=contador)
#decimatoma(tomas=contador)
onceavatoma(tomas=contador)

#pruebatoma()

###-----------TEST------##
#primeratoma(folder="testeyes",tomas=contador)
#segundatoma(folder="testeyes",tomas=contador)
#terceratoma(folder="testeyes",tomas=contador)
#cuartatoma(folder="testeyes",tomas=contador)
#quintatoma(folder="testeyes",tomas=contador)
#sextatoma(folder="testeyes",tomas=contador)
#septimatoma(folder="testeyes",tomas=contador)
#octavatoma(folder="testeyes",tomas=contador)
#novenatoma(folder="testeyes",tomas=contador)
#decimatoma(folder="testeyes",tomas=contador)
#onceavatoma(folder="testeyes",tomas=contador)