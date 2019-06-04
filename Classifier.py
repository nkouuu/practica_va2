from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import cv2
from utils import prepareImage, getHOGVector, vectorToList
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

        print(self.labels)
        lda = LDA(n_components=2)
        samples_list = vectorToList(self.samples)
        print(samples_list)
        reduced_data = lda.fit(samples_list, self.labels).transform(samples_list)

        print('adsbfiubsdfuasbf', reduced_data)

    def classify(self,path):
        folders = path.split("/")
        label = int(folders[len(folders)-1])
        for imagePath in os.listdir(path):
            print(imagePath)
            img = cv2.imread(path+"/"+imagePath, 1)
            prep_img = prepareImage(img)
            self.samples.append(getHOGVector(prep_img))

        self.labels.append(label)
