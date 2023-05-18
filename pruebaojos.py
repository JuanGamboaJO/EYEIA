import face_recognition
import numpy as np
from maxMin import maxAndMin
from matplotlib import pyplot as plt
import cv2 as cv
import copy

def webcam(feed=True):
    webcam = cv.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0,0), fy=.15, fx=.15)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)

        feats=face_recognition.face_landmarks(smallframe)
        haventfoundeye = True

        if len(feats)>0:
            leBds,leCenter = maxAndMin(feats[0]['left_eye'],mult = 1/.15)

            left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(left_eye, 50, 255, 0)
            TMP = 255 - np.copy(thresh)
            y = np.sum(TMP, axis=1)
            x = np.sum(TMP, axis=0)
            y = y / len(TMP[0])
            x = x / len(TMP)
            y = y > np.average(y) + np.std(y)#*1.2
            x = x > np.average(x) + np.std(x)#*1.2

            try:
                y = int(np.dot(np.arange(1, len(y) + 1), y) / sum(y))
            except:
                y = int(np.dot(np.arange(1, len(y) + 1), y) / 1)

            try:
                x = int(np.dot(np.arange(1, len(x) + 1), x) / sum(x))
            except:
                x = int(np.dot(np.arange(1, len(x) + 1), x) / 1)
            
            
            haventfoundeye = False
        
            left_eye = cv.cvtColor(left_eye, cv.COLOR_GRAY2BGR)
            cv.circle(left_eye, (x, y), 2, (20, 20, 120), 3)  
            cv.circle(left_eye, (int(leCenter[0]), int(leCenter[1])), 2, (120, 20, 20), 3)

            if feed:
                cv.imshow('frame', left_eye)
                if cv.waitKey(1) & 0xFF == ord('q'):
                        break
            elif not haventfoundeye:
                    plt.imshow(left_eye)
                    plt.title('my EYEBALL')
                    plt.show()
                    return left_eye
            
webcam()