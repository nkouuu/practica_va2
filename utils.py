import cv2

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

def getHOGVector(img):
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((0, 0),)
    hog = cv2.HOGDescriptor()
    return hog.compute(img,winStride,padding,locations)

#def train()
img = prepareImage(img)
img = cv2.resize(img.copy(),(30,30),interpolation = cv2.INTER_AREA)

print(getHOGVector(img))



cv2.waitKey(0)