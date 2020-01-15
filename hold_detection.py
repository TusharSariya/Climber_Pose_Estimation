import cv2
import numpy as np
import math

def hsv2bgr(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return b, g, r

def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def highligh(image,color):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    saturation = (int)(255*0.30)
    value = (int)(0)
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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=5)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),iterations=5)
    #mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations = 5)

    #image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    contours, hierarchy =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    line_color = hsv2bgr(color*2,1,1)
    cv2.drawContours(image, contours, -1, line_color,2)
    viewImage(image) ## 5
    

image = cv2.imread('climbing_1.jpg')
viewImage(image)
image = cv2.pyrMeanShiftFiltering(image,15,30)
viewImage(image)

#viewImage(image)
red = 0
yellow = 20
green = 65
blue = 100
purple = 120
black = 0

highligh(image, red)
highligh(image, yellow)
highligh(image, green)
highligh(image, blue)
highligh(image, purple)