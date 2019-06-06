import sys
import subprocess
import os
import cv2
import argparse
from lda import LDA
from utils import equalize
from Classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--train')
parser.add_argument('--detector')


def main():
    arguments = parser.parse_args()
    train_path = "train_recortadas"
    test_path = "test_reconocimiento"
    classifier ="lda"
    if(arguments.train):
        arguments.train
    if (arguments.detector):
        print('Detectado parametro classifier.\nSolo esta disponible el classifier por defecto.')
        classifier = arguments.classifier
    elif(arguments.test):
        test_path = arguments.test
    print("The script has " +str(3)+' arguments:')
    print("Train path: " +str(train_path)+'.')
    print("Test path: " +str(test_path)+'.')
    print("Clasificador: " +str(classifier))
    #imagenes = os.listdir(test_path)

    try:
        trainClassesPath = []
        for f in os.listdir(train_path):
            trainClassesPath.append(train_path+"/"+f)
        cl = Classifier([]).start(trainClassesPath, test_path)
        
    except Exception as e:
        print('Algo ha ido mal, por favor comprueba que tienes la version 3.6.x de Python y la version 3.x de OpenCV')
        print(str(e))
    print('Deteccion finalizada.')

main()
