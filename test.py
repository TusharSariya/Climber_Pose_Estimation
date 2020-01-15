
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
    yellow = 20
    green = 65
    blue = 100
    purple = 120
    black = 0
    colors = [red,yellow,green,blue,purple]

    image = cv2.imread('climbing_1.jpg')
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
            upper = np.array([variance,255,255])
            mask1 = cv2.inRange(hsv, lower, upper)
            
            # Range for upper range
            lower = np.array([180-variance,saturation,value])
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
        mask_final = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=5)
        masks.append(mask_final)
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
            #cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            #cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            #cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            i = 0
    return points

def det_climbing_route(image,hold_masks,pose_points,colors):
    i = 0
    for mask in hold_masks:
        contours, hierarchy =  cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        line_color = hsv2bgr(colors[i]*2,1,1)
        cv2.drawContours(image, contours, -1, line_color,2)
        i = i + 1
    for point in pose_points:
        cv2.circle(image, point, 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    return image

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