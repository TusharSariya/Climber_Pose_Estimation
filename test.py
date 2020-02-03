
#input image
#apply manipulation for more clear image
#detect holds for each color
#apply pose detection
#remove torso and limbs from detection
#find overlap between hands and hold

import cv2
import numpy as np
import math
import time

def main():
    #hsv color of holds
    red = 0
    yellow = 25
    green = 70
    blue = 100
    purple = 122.5
    black = 0
    colors = [red,yellow,green,blue,purple]

    image = cv2.imread('climbing_2.jpg')
    manipulate(image)
    hold_masks = det_holds(image,colors)
    pose_points = det_pose(image)
    #rem_climber(image,hold_masks,pose_points)
    image = det_climbing_route(image,hold_masks,pose_points,colors)
    viewImage(image)

def manipulate(image):
    image = cv2.pyrMeanShiftFiltering(image,15,30)

def det_holds(image,colors):
    

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    saturation = (int)(255*0.30)
    value = (int)(0)
    variance = 10
    masks = []
    for color in colors:
        if(color == 0):
            # Range for lower red
            
            lower = np.array([0,saturation,value])
            upper = np.array([5,255,255])
            mask1 = cv2.inRange(hsv, lower, upper)
            
            # Range for upper range
            lower = np.array([180-5,saturation,value])
            upper = np.array([180,255,255])
            mask2 = cv2.inRange(hsv,lower,upper)
            
            # Generating the final mask to detect red color
            mask = mask1+mask2
        elif(color == 99):
            #black
            i = 0
        else:
            lower = np.array([color-variance,saturation,value])
            upper = np.array([color+variance,255,255])
            mask = cv2.inRange(hsv, lower, upper)

        # Refining the mask corresponding to the detected red color
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
        masks.append(mask)
        #apply contours
        #contours, hierarchy =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #line_color = hsv2bgr(color*2,1,1)
        #cv2.drawContours(image, contours, -1, line_color,2)
        #viewImage(image) ## 5
    return masks

def det_pose(image):
    MODE = "MPI"

    if MODE is "COCO":
        protoFile = "pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif MODE is "MPI" :
        protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    
    frame = image
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            points.append((int(x), int(y)))
        else :
            points.append(None)
    return points

#this was actually insane
def det_climbing_route(image,hold_masks,pose_points,colors):
    iter = 0
    hold_rect = []
    hold_rect.append([])

    #iterate through all masks
    for mask in hold_masks:
        contours, hierarchy =  cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:
                filtered_contours.append(contour)

        #find max and min values of contour
        #iterate through each contour 
        for contour in filtered_contours:
            x_max = 0
            y_max = 0
            x_min = 3000
            y_min = 3000
            #iterate through points in contour
            for point in contour:
                point = point[0:1][0][0:2]
                if point[0] < x_min : x_min = point[0]
                if point[0] > x_max : x_max = point[0]
                if point[1] < y_min : y_min = point[1]
                if point[1] > y_max : y_max = point[1]
            '''extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            extBot = tuple(contour[contour[:, :, 1].argmax()][0])'''
            #iterate through all torso and knees and delete if overlap
            for p in [2,5,8,9,11,12,14]:
                point = pose_points[p]
                #delete contour of overlap with pose
                if(point[0]>x_min and point[0]<x_max and point[1]>y_min and point[1]<y_max):
                    #cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,0,255),1)
                    index = get_index(contour,filtered_contours)
                    del filtered_contours[index]
                    break
        line_color = hsv2bgr(colors[iter]*2,1,1)
        cv2.drawContours(image, filtered_contours, -1, line_color,2)
        iter += 1
    #4.7.10.13
    for point in pose_points:
        #check for surrounding holds
        cv2.circle(image, point, 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    return image

#https://stackoverflow.com/questions/53065245/test-if-a-numpy-array-is-a-member-of-a-list-of-numpy-arrays-and-remove-it-from
def get_index(array, list_of_arrays):
    for j, a in enumerate(list_of_arrays):
        if np.array_equal(array, a):
            return j
    return None


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

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

if __name__ == "__main__":
    main()
