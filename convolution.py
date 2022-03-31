import cv2
import numpy as np


def resizeDown(img):
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img, (int(w * 0.8), int(h * 0.8)))
    return img


def resizeUp(img, h=960, w=1280):
    img = cv2.resize(img, (h, w))
    return img


# Свертки для выявления границ на изображении
def conv(path, h, w):
    image = cv2.imread("sobel.jpg")
    image = resizeDown(image)

    kernel = np.array([
        [-2, 0, 2, 0, -2],
        [0, -1, 1, -1, 0],
        [2, 1, 1, 1, 2],
        [0, -1, 1, -1, 0],
        [-2, 0, 2, 0, -2]
    ])
    img1 = cv2.filter2D(image, -1, kernel)
    img1 = resizeUp(img1, h, w)

    kernel2 = np.array([
        [0, 0, -2, 0, 0],
        [0, 0, -1, 0, 0],
        [-2, -1, 13, -1, -2],
        [0, 0, -1, 0, 0],
        [0, 0, -2, 0, 0]
    ])
    img2 = cv2.filter2D(image, -1, kernel2)
    img2 = resizeUp(img2, h, w)

    kernel2 = np.array([
        [0, 0, -2, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 4, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, -2, 0, 0]
    ])
    img3 = cv2.filter2D(image, -1, kernel)
    img3 = resizeUp(img3, h, w)
    # Соединяем результаты
    conv_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 1)
    conv_img = cv2.addWeighted(conv_img, 0.7, img3, 0.3, 1)
    image = cv2.imread(path)
    new_img = cv2.addWeighted(conv_img, 0.9, image, 0.1, 1)
    cv2.imwrite('conv.jpg', new_img)


