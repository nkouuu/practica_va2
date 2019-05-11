import sys
import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--train')
parser.add_argument('--detector')


def main():
    arguments = parser.parse_args()
    train_path = "train"
    test_path = "test_corto"
    detector ="detector.py"
    if(arguments.train):
        arguments.train
    if (arguments.detector):
        print('Detectado parametro detector.\nSolo esta disponible el detector por defecto.')
        #detector = arguments.detector
    elif(arguments.test):
        test_path = arguments.test
    print("The script has " +str(arguments)+' arguments:')
    print("Train path: " +str(train_path)+'.')
    print("Test path: " +str(test_path)+'.')
    print("Detector: " +str(detector))
    imagenes = os.listdir(test_path)

    try:
        #Aplicamos el detector a cada imagen , pero sin pasarle el train_path de momento
        for img in imagenes:
            subprocess.call(["python",str(detector), str(test_path)+"/"+str(img)])
        #p = subprocess.call(["python3",str(detector), 'train/00004.ppm'])
    except:
        print('Algo ha ido mal, por favor comprueba que tienes la version 3.6.x de Python y la version 3.x de OpenCV')
    print('Deteccion finalizada.')


main()