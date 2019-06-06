import cv2
import numpy as np


# Equalizacion de la imagen usando un equalizado adaptativo
def equalize(img):
    cpy = img.copy()
    grey = cv2.cvtColor(cpy, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe = eq.apply(grey)
    return clahe

# Tratamiento y escalado de la imagen
def prepareImage(img):
    img_cpy = img.copy()
    image = equalize(img_cpy)
    image = cv2.resize(image,(30,30),interpolation = cv2.INTER_LINEAR)
    return image


def reshapeList(list):
    result_list = np.array(list)

    # Tupla con las dimensiones del array
    rows, columns, i = result_list.shape
    return result_list.reshape(rows, columns)


def getHOGVector(img):
    # Tiene que cumplir (winSize.width - blockSize.width) % blockStride.width == 0 && (winSize.height - blockSize.height) % blockStride.height == 0
    hog = cv2.HOGDescriptor(_winSize=(30,30),_blockSize=(15,15),_blockStride=(5,5),_cellSize=(5,5),_nbins=9)
    return hog.compute(img)


def writeInFile(file, list, results):
    file = open(file, 'w')
    for x in range(0, len(list)):
        file.write(f'{list[x]} ; {("","0")[results[x] < 10]}{str(results[x])} \n')
    file.close()





