import cv2
import numpy as np

# Equalizacion de la imagen usando un equalizado adaptativo
def equalize(img):
    cpy = img.copy()
    grey = cv2.cvtColor(cpy, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe = eq.apply(grey)
    return clahe
    #cv2.waitKey(0)

img = cv2.imread("train_recortadas/00/00000.ppm",1)


def prepareImage(img):
    img_cpy = img.copy()
    image = cv2.resize(img_cpy,(30,30),interpolation = cv2.INTER_AREA)
    image = equalize(image)
    return image

def vectorToList(vector):

    result_list = np.array(vector)

    #Tupla con las dimensiones del array
    rows, columns, i = result_list.shape
    return result_list.reshape(rows, columns)

def getHOGVector(img):

    winSize= (30,30)
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((0, 0),)

    #Tiene que cumplir (winSize.width - blockSize.width) % blockStride.width == 0 && (winSize.height - blockSize.height) % blockStride.height == 0
    hog = cv2.HOGDescriptor(_winSize=(30,30),_blockSize=(15,15),_blockStride=(5,5),_cellSize=(5,5),_nbins=9)
    return hog.compute(img)

#def train()
img = prepareImage(img)
img = cv2.resize(img.copy(),(30,30),interpolation = cv2.INTER_AREA)

print(getHOGVector(img))



cv2.waitKey(0)
