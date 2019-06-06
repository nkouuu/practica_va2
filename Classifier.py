from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, average_precision_score
import cv2
from utils import prepareImage, getHOGVector, reshapeList
import os
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self,classifier):
        self.train_samples = []
        self.train_labels = []
        self.test_samples = []
        self.test_labels = []
        self.classifier = classifier
        print('Clasifier initialized!')


    def start(self,train_classes_folders, test_folder):
        for folderPath in train_classes_folders:
            self.classify(folderPath, "train")

        self.classify(test_folder, "test")
        self.classifier = LDA()

        # Reducir dimensionalidad
        reduced_data = self.reduce_dimensionality(self.train_samples, self.train_labels)

        # Train classifier
        lda_train_result = self.train_classifier(reduced_data)

        # Obtener la precision de la prediccion
        train_accuracy = self.get_accuracy(lda_train_result, self.train_labels);
        print("Precisión de la predicción del entrenamiento: ", train_accuracy)

        # Test
        test_samples_list = self.prepare_test(self.test_samples)
        lda_test_result = self.classifier.predict(test_samples_list)
        test_accuracy = self.get_accuracy(lda_test_result, self.test_labels);
        print("Precisión de la predicción del test: ", test_accuracy)

        


    def classify(self,path, type):
        if (type == "train"):
            folders = path.split("/")
            label = int(folders[len(folders)-1])
            for imagePath in os.listdir(path):
                img = cv2.imread(path+"/"+imagePath, 1)
                prep_img = prepareImage(img)
                self.train_samples.append(getHOGVector(prep_img))
                self.train_labels.append(label)
        elif (type == "test"):
            for imagePath in os.listdir(path):
                if (imagePath != ".directory"): #Error al leer imagenes
                    label = int(imagePath[:2])
                    img = cv2.imread(path+"/"+imagePath, 1)
                    prep_img = prepareImage(img)
                    self.test_samples.append(getHOGVector(prep_img))
                    self.test_labels.append(label)

    def reduce_dimensionality(self, samples, labels):
        #Es necesario utilizar np.reshape ya que sklearn requiere datos de forma (row number, column number).
        samples_list = reshapeList(samples)

        reduced_data = self.classifier.fit(samples_list, labels).transform(samples_list)
        #HOG es float 32 y el transform de lda es float 64
        return reduced_data.astype(np.float32)

    def train_classifier(self, reduced_data):
        self.classifier.fit(reduced_data, self.train_labels)
        return self.classifier.predict(reduced_data)

    def get_accuracy(self, samples, labels):
        return accuracy_score(np.array(labels), samples)
    
    def prepare_test(self, samples):
        result = reshapeList(samples)
        print('in')
        result = self.classifier.transform(result)
        return result.astype(np.float32)
