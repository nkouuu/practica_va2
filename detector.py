import cv2

from utils import applyRedMask, detectSignals

# Para un path de una imagen devolvemos la mascara roja de la imagen redimensionada
def getResizedMask(imgPath):
    img = cv2.imread(imgPath, 1)
    img = cv2.resize(img, (25, 25))
    mask, _ = applyRedMask(img)
    return mask

def getSignalsFromImg(imgPath):
    # mascaras rojas por defecto
    prohibidoMask = getResizedMask('prohibido.png')
    peligroMask = getResizedMask('peligro.jpg')
    stopMask = getResizedMask('stop.jpg')


    imgRectangles = []
    imagen = cv2.imread(imgPath,1)
    imagen_copy = imagen.copy()


    # pasamos imagen a escala de grises
    grey = cv2.cvtColor(imagen_copy, cv2.COLOR_BGR2GRAY)

    # Aplicamos el threshold para eliminar ruidos truncando los pixeles dentro de un rango de blanco
    ret, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_TRUNC)

    # Creamos un detector de regiones de alto contraste
    mser = cv2.MSER_create()

    # Obtenemos las regiones detectadas y las transformamos en rectangulos
    regions, _ = mser.detectRegions(thresh)
    boundingRectangles = [cv2.boundingRect(p) for p in regions]

    # Por cada rectangulo
    for index, rectangle in enumerate(boundingRectangles):
        increase = 5
        x, y, w, h = rectangle
        crop_img = imagen_copy[y:y + h, x:x + w]
        crop_img = cv2.resize(crop_img, (30, 30))
        difference = w / h
        # Descartamos las regiones que no sean lo suficientemente cuadradas
        if (difference > 0.7 and difference < 1.3):
            # cv2.rectangle(cimg,(x-increase,y-increase),(x+w+increase,y+h+increase),(0,255,0),2)
            imgRectangles.append([x - increase, y - increase, w + increase, h + increase])
            #cv2.imshow("imagen."+str(index),imagen_copy[y:h,x:w])
    res,res_por_tipo,imgs_resultado = detectSignals(imgRectangles, imagen_copy, [prohibidoMask, peligroMask, stopMask])
    print(str(len(imgs_resultado))+" Señales detectadas en la imagen.\n" )
    return imgs_resultado

