import cv2
import numpy as np

def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image = cv2.imread('climbing_1.jpg')
viewImage(image)
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
viewImage(hsv)
RGB = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
viewImage(RGB)