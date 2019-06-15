import sys
import subprocess
import os
import cv2
import argparse
import shutil
import matplotlib as plt
from detector import getSignalsFromImg
from utils import writeInFile
from Classifier import Classifier
from Graphics import  Graphics

parser = argparse.ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--train')
parser.add_argument('--classifier')
parser.add_argument('--img_path')

def main():
    arguments = parser.parse_args()
    train_path = "train_recortadas"
    test_path = "test_reconocimiento"
    classifier ="LDA-BAYES"
    imgPath = None
    if(arguments.train):
        train_path = arguments.train
    if (arguments.classifier):
        print('Detectado parametro classifier: '+arguments.classifier)
        classifier = arguments.classifier
    if(arguments.test):
        test_path = arguments.test
    if (arguments.img_path):
        imgPath = arguments.img_path
    print("The script has " +str(3)+' arguments:')
    print("Train path: " +str(train_path)+'.')
    print("Test path: " +str(test_path)+'.')
    print("Clasificador: " +str(classifier))
    #imagenes = os.listdir(test_path)

    try:
        trainClassesPath = []
        for f in os.listdir(train_path):
            trainClassesPath.append(train_path+"/"+f)
        cl = Classifier(classifier)
        classifier_result, test_img_names, test_labels, test_accuracy, train_accuracy = cl.start(trainClassesPath, test_path)

        # Si hay una imagen real para analizar
        if(imgPath is not None):
            print("Detectada imagen adicional para analizar. \nComenzando deteccion de señales...")
            # Obtenemos las señales de la imagen y las guardamos en una carpeta para procesarlas
            imgs = getSignalsFromImg(imgPath)
            folder = './temporary_test'
            if not os.path.exists(folder):
                os.makedirs(folder)
            else:
                shutil.rmtree(folder)  # removes all the subdirectories!
                os.makedirs(folder)
            for i, img in enumerate(imgs):
                im = plt.image.imsave(folder+"/image"+str(i)+".png",cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Una vez guardadas las señales reconocidas de la imagen procedemos a clasificarlas --> ToDo

        # Graficos
        graphics = Graphics()
        graphics.accuracy_graphic(train_accuracy, "Entrenamiento " + classifier)
        graphics.accuracy_graphic(test_accuracy, "Test " + classifier)
        graphics.conf_matrix(classifier_result, test_labels)
        graphics.get_f1_score(classifier_result, test_labels)

        # Escribir resultados
        writeInFile("resultado.txt", test_img_names, classifier_result)
        
    except Exception as e:
        print('Algo ha ido mal, por favor comprueba que tienes la version 3.6.x de Python y la version 3.x de OpenCV')
        print(str(e))
    print('Reconocimiento finalizado. Resultados escritos en resultados.txt')

main()
