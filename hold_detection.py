import cv2
import numpy as np
import random

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def highligh(image,color):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = 255
    saturation = (int)(255*0.40)
    value = (int)(255*0.30)
    variance = 10
    #check if red
    if(color == red):
        # Range for lower red
        
        lower = np.array([0,saturation,value])
        upper = np.array([variance,255,255])
        mask1 = cv2.inRange(hsv, lower, upper)
        
        # Range for upper range
        lower = np.array([180-variance,saturation,value])
        upper = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower,upper)
        
        # Generating the final mask to detect red color
        mask = mask1+mask2
    else:
        lower = np.array([color-variance,saturation,value])
        upper = np.array([color+variance,255,255])
        mask = cv2.inRange(hsv, lower, upper)

    # Refining the mask corresponding to the detected red color
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=5)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),iterations=5)
    #mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations = 5)

    #image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    contours, hierarchy =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,0,0),1)
    viewImage(image) ## 5
    

image = cv2.imread('climbing_3.jpg')
viewImage(image)
#hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#hsv[:,:,2] = 255
#image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#viewImage(image)
red = 0
yellow = 20
green = 65
blue = 100
purple = 125
black = 0

highligh(image, red)
highligh(image, yellow)
highligh(image, green)
highligh(image, blue)
highligh(image, purple)