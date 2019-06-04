import cv2

# Equalizacion de la imagen usando un equalizado adaptativo
def equalize(img):
    cpy = img.copy()
    grey = cv2.cvtColor(cpy, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe = eq.apply(grey)
    return clahe
    #cv2.waitKey(0)

img = cv2.imread("train_recortadas/00/00000.ppm",0)


def prepareImage(img):
    img_cpy = img.copy()
    image = cv2.resize(img_cpy,(30,30))
    image = equalize(image)
    return image

def getHOGVector(img):
    hog = cv2.HOGDescriptor()
    return hog.compute(img)

#img = prepareImage(img)

print(getHOGVector(img))
cv2.waitKey(0)