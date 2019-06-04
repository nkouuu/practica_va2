from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import cv2
import numpy as np
from utils import prepareImage,getHOGVector
import os

class Classifier:
    def __init__(self,classifier):
        self.samples = []
        self.labels = []
        self.classifier = classifier
        print('Clasifier initialized!')


    def start(self,clasesFolders):
        for folderPath in clasesFolders:
            self.classify(folderPath)
        print(self.samples)
        print(self.labels)
    def classify(self,path):
        folders = path.split("/")
        label = folders[len(folders)-1]
        for imagePath in os.listdir(path):
            print(imagePath)
            img = cv2.imread(path+"/"+imagePath, 1)
            prep_img = prepareImage(img)
            self.samples.append(getHOGVector(prep_img))

        self.labels.append(label)
