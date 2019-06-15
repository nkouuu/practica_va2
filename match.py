import cv2
import numpy as np

#Comprobamos si la imagen contiene
def detect(img):
    cimg =img.copy()
    shape = None
    grey = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    #Obtenemos contornos de la imagen, que seran los candidatos a ser figuras geometricas
    _, contours, _ = cv2.findContours(
        grey, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #obtenemos el area de cada contorno
    areas = [cv2.contourArea(c) for c in contours]
    i = 0
    for area in areas:
        #ignoramos areas muy pequeñas
        if area > 300:
            contorno = contours[i]
            #Aproximamos el contorno con un 4% de error y obtenemos los vertices
            approx = cv2.approxPolyDP(contorno, 0.04 * cv2.arcLength(contorno, True), True)
            # con 3 vertices tenemos un triangulo
            if len(approx) == 3:
                shape = "triangle"
            # para 4 vertices
            elif len(approx) == 4:

                # transformamos a rectangulo y comprobamos relacion ancho/alto
                (x, y, w, h) = cv2.boundingRect(approx)
                wh = w / float(h)

                # si wh es cercano a 1 significa que los lados son casi iguales, por lo que es un cuadrado,
                # sino es un rectangulo
                shape = "square" if wh >= 0.95 and wh <= 1.05 else "rectangle"


            # si tiene mas de 5 vertices damos por hecho que es un pentagono
            else:
                shape = "pentagon"
        i += 1

    return shape


#con la transformada de Hough comprobamos si la imagen contiene circulos
def detectCircles(img):
    #x1 < x < x2 and y1 < y < y2
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    if(circles is None or len(circles) == 0):
        return
    else:
        circles = np.uint16(np.around(circles))
    return circles
