from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, average_precision_score
import cv2
from utils import prepareImage, getHOGVector, reshapeList
import os
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self,classifier):
        self.samples = []
        self.labels = []
        self.classifier = classifier
        print('Clasifier initialized!')


    def start(self,clasesFolders):
        for folderPath in clasesFolders:
            self.classify(folderPath)

        self.classifier = LDA()

        # Reduce dimensionality
        reduced_data = self.reduce_dimensionality(self.samples, self.labels)

        # Train classifier
        lda_result = self.train_classifier(reduced_data)
        print(lda_result)

        # Get prediction accuracy
        train_accuracy = self.get_accuracy(self.labels, lda_result);
        print("Precisión de la predicción del entrenamiento: ", train_accuracy)


    def classify(self,path):
        folders = path.split("/")
        label = int(folders[len(folders)-1])
        for imagePath in os.listdir(path):
            print(imagePath)
            img = cv2.imread(path+"/"+imagePath, 1)
            prep_img = prepareImage(img)
            self.samples.append(getHOGVector(prep_img))
            self.labels.append(label)

    def reduce_dimensionality(self, samples, labels):
        #Es necesario utilizar np.reshape ya que sklearn requiere datos de forma (row number, column number).
        samples_list = reshapeList(samples)
        print(samples_list)

        reduced_data = self.classifier.fit(samples_list, labels).transform(samples_list)
        #HOG es float 32 y el transform de lda es float 64
        return reduced_data.astype(np.float32)

    def train_classifier(self, reduced_data):
        self.classifier.fit(reduced_data, self.labels)
        return self.classifier.predict(reduced_data)

    def get_accuracy(self, samples, labels):
        return accuracy_score(np.array(labels), samples)
