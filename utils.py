import cv2
import numpy as np
from match import detectCircles, detect

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



# aplicar mascara de rojos a uma imagen
def applyRedMask(img):
    # pasamos imagen a hsv
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)

    # definimos tanto la mascara de rojos bajos como altos y las sumamos
    # rojos bajos (0-12)
    lower_red = np.array([0, 50, 75], dtype=np.uint8)
    upper_red = np.array([12, 255, 255], dtype=np.uint8)
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # rojos altos (170-180)
    lower_red2 = np.array([170, 50, 75], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_red2, upper_red2)

    # sumamos las mascaras
    mask = mask0 + mask1

    # pintamos de blanco los pixeles rojos y negro el resto
    output_hsv = hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    output_hsv[np.where(mask != 0)] = 255

    # devolvemos tanto la imagen con la mascara aplicada como el numero de pixeles rojos que contiene
    return output_hsv, len(output_hsv[np.where(mask != 0)])


# funcion que aplica brillo o contraste a una imagen
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# Funcion que a partir de las regiones detectadas y filtradas comprueba si hay match con alguna señal
def detectSignals(imgRectangles, img, masks):
    cimg = img.copy()
    max_score = 0
    shape = ''
    resultado = []
    resultado_por_tipo = []
    imgs_resultado = []
    for index, rectangle in enumerate(imgRectangles):
        x, y, w, h = rectangle
        if (x > 0 and y > 0 and w > 0 and h > 0):
            crop_img = cimg[y:y + h, x:x + w]
            crop_img = cv2.resize(crop_img, (25, 25))
            output_hsv, numRojos = applyRedMask(crop_img)

            # Comprobar prohibido
            resProhibido = cv2.matchTemplate(output_hsv, masks[0], cv2.TM_CCORR_NORMED)
            threshold = 0.1
            locProhibido = np.where(resProhibido >= threshold)

            # Comprobar peligro
            resPeligro = cv2.matchTemplate(output_hsv, masks[1], cv2.TM_CCORR_NORMED)

            locPeligro = np.where(resPeligro >= 1)

            # Comprobar STOP
            resStop = cv2.matchTemplate(output_hsv, masks[2], cv2.TM_CCORR_NORMED)

            locStop = np.where(resStop >= 1)

            # Nos quedamos con la maxima coincidencia
            max = np.maximum(resPeligro, resStop)
            max = np.maximum(max, resProhibido)
            _, max_val, _, max_loc = cv2.minMaxLoc(max)
            score = int(max_val * 100)

            if (score > max_score): max_score = score

            # Comprobamos si hay alguna figura geometrica
            shape = detect(crop_img)

            #definimos un umbral
            if(score>20):
                # Por cada tipo de señal miramos si coincide la figura geometrica y añadimos a las listas de resultados los datos
                if (max == resPeligro and shape == 'triangle'):
                    imgs_resultado.append(crop_img)
                    resultado.append(str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";1;" + str(score))
                    resultado_por_tipo.append(
                        str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";" + "2" + ";" + str(score))
                    cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 255, 0), 2)

                elif (max == resProhibido):
                    circles = detectCircles(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY))
                    if (circles is not None):
                        imgs_resultado.append(crop_img)
                        resultado.append(str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";1;" + str(score))
                        resultado_por_tipo.append(
                            str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";" + "1" + ";" + str(score))
                        cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 255, 0), 2)

                elif (max == resStop and shape == 'pentagon'):
                    imgs_resultado.append(crop_img)
                    resultado.append(str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";1;" + str(score))
                    resultado_por_tipo.append(
                        str(x) + ";" + str(y) + ";" + str(w) + ";" + str(h) + ";" + "3" + ";" + str(score))
                    cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return resultado, resultado_por_tipo,np.array(imgs_resultado)


