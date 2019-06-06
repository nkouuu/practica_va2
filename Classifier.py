from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import cv2
from utils import prepareImage, getHOGVector, reshapeList
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB



class Classifier:
    def __init__(self, classifier):
        self.train_samples = []
        self.train_labels = []
        self.test_samples = []
        self.test_labels = []
        if classifier == 'LDA-BAYES':
            self.classifier = LDA()
            self.lda = LDA()
        elif classifier == 'PCA-BAYES':
            self.classifier = PCA()
            self.bayes_classifier = GaussianNB()
        self.classifier_type = classifier
        self.test_img_names = []
        print('Clasifier initialized!')

    def start(self, train_classes_folders, test_folder):
        for folderPath in train_classes_folders:
            self.classify(folderPath, "train")

        self.classify(test_folder, "test")


        # Reducir dimensionalidad
        reduced_data = self.reduce_dimensionality(self.train_samples, self.train_labels)

        # Train classifier
        if self.classifier_type == 'LDA-BAYES':
            train_result = self.train_classifier(reduced_data, self.lda)
        elif self.classifier_type == 'PCA-BAYES':
            train_result = self.train_classifier(reduced_data, self.bayes_classifier)

        # Obtener la precision de la prediccion
        train_accuracy = self.get_accuracy(train_result, self.train_labels);
        print(f'Precisión de la predicción del entrenamiento: {train_accuracy} - {train_accuracy*100}%')

        # Test
        test_samples_list = self.prepare_test(self.test_samples)
        if self.classifier_type == 'LDA-BAYES':
            test_result = self.lda.predict(test_samples_list)
        elif self.classifier_type == 'PCA-BAYES':
            test_result = self.bayes_classifier.predict(test_samples_list)
        
        test_accuracy = self.get_accuracy(test_result, self.test_labels);
        print(f'Precisión de la predicción del test: {test_accuracy} - {"{0:.2f}".format(test_accuracy*100)}%')

        return test_result, self.test_img_names, self.test_labels, test_accuracy*100, train_accuracy*100

    def classify(self,path, type):
        if type == "train":
            folders = path.split("/")
            label = int(folders[len(folders)-1])
            for imagePath in os.listdir(path):
                img = cv2.imread(path+"/"+imagePath, 1)
                prep_img = prepareImage(img)
                self.train_samples.append(getHOGVector(prep_img))
                self.train_labels.append(label)
        elif type == "test":
            for imagePath in os.listdir(path):
                if imagePath != ".directory":  # Error al leer imagenes
                    self.test_img_names.append(imagePath)
                    label = int(imagePath[:2])
                    img = cv2.imread(path+"/"+imagePath, 1)
                    prep_img = prepareImage(img)
                    self.test_samples.append(getHOGVector(prep_img))
                    self.test_labels.append(label)

    def reduce_dimensionality(self, samples, labels):
        # Es necesario utilizar np.reshape ya que sklearn requiere datos de forma (row number, column number).
        samples_list = reshapeList(samples)
        if self.classifier_type == 'LDA-BAYES':
            reduced_data = self.classifier.fit_transform(samples_list, np.array(labels))
        elif self.classifier_type == 'PCA-BAYES':
            reduced_data = self.classifier.fit_transform(samples_list)

        # HOG es float 32 y el transform de lda es float 64
        return reduced_data.astype(np.float32)

    def train_classifier(self, reduced_data, classifier):
        classifier.fit(reduced_data, np.array(self.train_labels))
        return classifier.predict(reduced_data)

    def get_accuracy(self, samples, labels):
        return accuracy_score(np.array(labels), samples)
    
    def prepare_test(self, samples):
        result = reshapeList(samples)
        result = self.classifier.transform(result)
        return result.astype(np.float32)
