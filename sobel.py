import cv2
import numpy as np


def adjust_gamma(image, gamma=2.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def sobel(path='test.jpg'):
    img = cv2.imread(path)
    img = adjust_gamma(img)
    h = img.shape[0]
    w = img.shape[1]

    img = cv2.resize(img, (int(w*0.7), int(h*0.7)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5, scale=1)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5, scale=1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    blur = cv2.blur(edge, ksize=(3, 3))

    new_img = cv2.addWeighted(gray, 0.07, blur, 0.93, 0.6)
    cv2.imwrite('sobel.jpg', new_img)
