import cv2
import numpy as np


# Функция, непосредственно выделяющая контур листа
def detect(path):
    # Читаем предобработанное изображение
    img = cv2.imread('conv.jpg')

    h = img.shape[0]
    w = img.shape[1]

    img = cv2.resize(img, (int(w*0.7), int(h*0.7)))

    # Переводим в чб
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Размытие
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_blur = cv2.GaussianBlur(img_blur, (3, 3), 0)

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Контур с наибольшей площадью и есть лист
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    cv2.drawContours(img, contours, max_index, (0, 255, 0), 3)
    #cv2.imshow('result', img)
    #cv2.waitKey(0)
    # Борьба с артефактами - обогнем весь контур и уменьшим количество звеньев
    hull = cv2.convexHull(cnt)
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    img = cv2.imread(path)
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.resize(img, (int(w*0.7), int(h*0.7)))
    cv2.drawContours(img, [approx], -1, (0, 255, 255), 4)
    # Сохраняем изображение в result.jpg и показываем пользователю
    cv2.imshow('result', img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
