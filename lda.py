from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import cv2
import numpy as np

class LDA:
    def __init__(self):
        print('LDA initialized!')


    def classify(self,img):
        eqImg = cv2.equalizeHist(img)