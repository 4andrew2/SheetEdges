import cv2
import numpy as np
import detect
#import utils
import sobel
import convolution


if __name__ == "__main__":
    path = str(input())
    sobel.sobel(path)
    w, h, _ = cv2.imread(path).shape
    convolution.conv(path, h, w)
    detect.detect(path)

