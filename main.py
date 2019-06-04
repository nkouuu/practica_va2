import sys
import subprocess
import os
import cv2
import argparse
from lda import LDA
from utils import equalize

parser = argparse.ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--train')
parser.add_argument('--detector')


def main():
    arguments = parser.parse_args()
    train_path = "train"
    test_path = "test"
    classifier ="lda"
    if(arguments.train):
        arguments.train
    if (arguments.detector):
        print('Detectado parametro classifier.\nSolo esta disponible el classifier por defecto.')
        classifier = arguments.classifier
    elif(arguments.test):
        test_path = arguments.test
    print("The script has " +str(arguments)+' arguments:')
    print("Train path: " +str(train_path)+'.')
    print("Test path: " +str(test_path)+'.')
    print("Clasificador: " +str(classifier))
    imagenes = os.listdir(test_path)

    try:
        #Aplicamos el clasificador a cada imagen
        '''for img in imagenes:
            subprocess.call(["python",str(classifier), str(test_path)+"/"+str(img)])'''
        #p = subprocess.call(["python3",str(detector), 'train/00004.ppm'])
        
    except:
        print('Algo ha ido mal, por favor comprueba que tienes la version 3.6.x de Python y la version 3.x de OpenCV')
    print('Deteccion finalizada.')

main()