from calendar import c
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as pyp
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
import cv2
from PIL import Image
import PIL.ImageOps as IO
import os, time

x, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["A", "B", "C", "D", "E", "F", "G","H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X","Y", "Z"]
nClasses = len(classes)

xTrain, xTest, yTrain, yTest = tts(x, y, test_size = 2500)

xTrainSc = xTrain/255.0
xTestSc = xTest/255.0

LR = lr(solver = "saga", multi_class = "multinomial").fit(xTrainSc, yTrain)
yPre = LR.predict(xTestSc)
accuracy = accuracy_score(yTest, yPre)
print(accuracy*100)

cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomRight = (int(width/2 +56), int(height/2 + 56))

        cv2.rectangle(gray, upperLeft, bottomRight, (0,255,0), 2)
        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]

        # Converting cv2 to PIL format
        impil = Image.fromarray(roi)
        imageBW = impil.convert("L")
        imageBW_resized = imageBW.resize((28,28), Image.ANTIALIAS)

        imageInverted = IO.invert(imageBW_resized)
        pixelFilter = 20
        min_pixel = np.percentile(imageInverted, pixelFilter)

        imageScaled = np.clip(imageInverted - min_pixel, 0, 255)
        maxPixel = np.max(imageInverted)

        imageScaled = np.asarray(imageScaled)/maxPixel
        sample = np.array(imageScaled).reshape(1,784)
        testPre = LR.predict(sample)
        print(testPre)
        
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) and 0xff==ord("q"):
            break
    
    except:
        pass

cap.release()
cv2.destroyAllWindows()