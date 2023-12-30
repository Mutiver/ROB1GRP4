#!/usr/bin/env python

import cv2
import urllib
import numpy as np
import math
from time import sleep
from skimage import morphology
from itertools import groupby

def get_from_webcam():
    print "try fecth from webcam..."
    stream = urllib.urlopen('http://192.168.0.20/image/jpeg.cgi')
    bytes = ''
    bytes += stream.read(64500)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')

    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype = np.uint8), cv2.IMREAD_COLOR)
        return i

def get_from_file(filename):
    print "loading from file..."
    return cv2.imread(filename)

coords = []
done = True

def FindBoard():
    while(done):

        image = get_from_webcam()
        y = 40
        x = 110
        h = 75
        w = 85
        Crop_image = image[y:y+h, x:x+w]
        Size_image = cv2.resize(Crop_image, (400,400))

        cv2.imwrite('Test.jpg', Size_image)

        jpn = cv2.imread('Test.jpg') #Test.jpg

        im6 = 2.0*jpn[:,:,1]-jpn[:,:,0]-jpn[:,:,2]

        im6 = im6 - im6.min()

        im6 = ((im6/im6.max())*255).astype(np.uint8)
        cv2.imshow('test', im6)

        print(im6.shape)

        print(im6.dtype)



    #im6 = 2.0*Size_image[:,:,1]-Size_image[:,:,0]-Size_image[:,:,2]

    #im6 = im6 - im6.min()

    #im6 = ((im6/im6.max())*255).astype(np.uint8)
    #cv2.imshow('test', im6)

    #print(im6.shape)

    #print(im6.dtype)

    #im7 = ((im6 > -50) * 255).astype(np.uint8)

    #cv2.imshow('Cam',im7)

    #key = cv2.waitKey(1)

    #edges = cv2.Canny(im6,20,35)

        ret, thresh = cv2.threshold(im6,67,255,0)

        im4 = cv2.dilate(thresh,np.ones((3,3)))
        cv2.imshow("dilate",im4)

        cleaned = morphology.remove_small_objects(im4>20, min_size=552, connectivity=2)
        cv2.imshow("cleaned", (cleaned*255).astype(np.uint8))

        _, contours, hierarchy = cv2.findContours((cleaned*255).astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if(len(contours) == 10):
            break

    print(len(contours))
    print(hierarchy)

    coords = []
    for cnt,hie in zip(contours, hierarchy[0]):

        if hie[3] != -1:

            area = cv2.contourArea(cnt)
            circumference = cv2.arcLength(cnt, True)
            if 1 < 2:
            #</alter these lines>
                # Calculate the centre of mass
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                coords.append((cx,cy))

    #sort by x
    sorted_cords = sorted(coords, key=lambda x: x[0])
    print(sorted_cords)
    #take x-values in sets of 3, find the middlevalue and replace
    #first 3

    new_x1 = (sorted_cords[0][0] + sorted_cords[1][0] + sorted_cords[2][0])/3
    new_x2 = (sorted_cords[3][0] + sorted_cords[4][0] + sorted_cords[5][0])/3
    new_x3 = (sorted_cords[6][0] + sorted_cords[7][0] + sorted_cords[8][0])/3

    #updating the points
    new_x_values = [new_x1, new_x1, new_x1, new_x2, new_x2, new_x2, new_x3, new_x3, new_x3]
    for i in range(len(sorted_cords)):
        sorted_cords[i] = (new_x_values[i], sorted_cords[i][1])

    sorted_cords = sorted(sorted_cords,key=lambda x: (x[0], x[1]))

    print(coords)
    print(sorted_cords)
    print(sorted_cords[0])
    #test functions
    center = (int(sorted_cords[0][0]), int(sorted_cords[0][1]))

    image = cv2.circle(jpn, center, radius=0, color=(0, 0, 255), thickness=10)

    #cnt = contours[4]
    #im5 = cv2.drawContours(Size_image, [contours[4]], -1, (0,0,255),3)
    #im5 = cv2.drawContours(Size_image, [contours[1]], 0, (0,0,255), 3)
    #print(contours)
    coords = sorted_cords

    while(done):
        cv2.imshow('Cam',image)
        #cv2.imshow('Cam2',im5)
        #cv2.imwrite('Test.jpg', im6)
        
        key = cv2.waitKey(1)
    #return coords

test_coords = [(98.0, 86), (98.0, 204), (98.0, 320), (199.33333333333334, 86), (199.33333333333334, 204), (199.33333333333334, 320), (299.0, 86), (299.0, 203), (299.0, 317)]
position = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]
brickCoords = []
def FindBricks(coords):

    image = get_from_webcam()
    y = 40
    x = 110
    h = 75
    w = 85
    Crop_image = image[y:y+h, x:x+w]
    Size_image = cv2.resize(Crop_image, (400,400))

    cv2.imwrite('Test.jpg', Size_image)

    jpn = cv2.imread('Test.jpg') #Test.jpg

    blue_brickCoords = []
    red_brickCoords = []
    # Read the image
    frame = jpn  # Replace cam image


    red=frame[:,:,2] 
    green=frame[:,:,1] 
    blue=frame[:,:,0] 

    # blue 
    temp = cv2.addWeighted(blue, 0.4, green, -0.2, 50)
    exr = cv2.addWeighted(temp, 1, red, -0.2, 0)

    # Find places with an exr value over 60
    retval, thrsholdeded = cv2.threshold(exr, 60, 255, cv2.THRESH_BINARY)

    # show the thresholded image
    #cv2.imshow('exr', thrsholdeded)


    imcleaned = morphology.remove_small_objects(thrsholdeded>20, min_size=552, connectivity=2)
    _, contours, hierarchy = cv2.findContours((imcleaned*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    #cv2.imshow('cleaned', (imcleaned*255).astype(np.uint8))
    for cnt in contours:
   
        area = cv2.contourArea(cnt)
        circumference = cv2.arcLength(cnt, True)
        if 1 < 2:
        #</alter these lines>
            # Calculate the centre of mass
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            blue_brickCoords.append((cx,cy))

    for coord in blue_brickCoords:
    
        x_coord, y_coord = coord
        if coords[8][0] -25<x_coord<coords[8][0]+25 and coords[8][1]-25<y_coord<coords[8][1]+25:
            position[2][2] = 2
        elif coords[7][0]-25<x_coord<coords[7][0]+25 and coords[7][1]-25<y_coord<coords[7][1]+25:
            position[2][1] = 2
        elif coords[6][0]-25<x_coord<coords[6][0]+25 and coords[6][1]-25<y_coord<coords[6][1]+25:
            position[2][0] = 2
        elif coords[5][0]-25<x_coord<coords[5][0]+25 and coords[5][1]-25<y_coord<coords[5][1]+25:
            position[1][2] = 2
        elif coords[4][0]-25<x_coord<coords[4][0]+25 and coords[4][1]-25<y_coord<coords[4][1]+25:
            position[1][1] = 2
        elif coords[3][0]-25<x_coord<coords[3][0]+25 and coords[3][1]-25<y_coord<coords[3][1]+25:
            position[1][0] = 2
        elif coords[2][0]-25<x_coord<coords[2][0]+25 and coords[2][1]-25<y_coord<coords[2][1]+25:
            position[0][2] = 2
        elif coords[1][0]-25<x_coord<coords[1][0]+25 and coords[1][1]-25<y_coord<coords[1][1]+25:
            position[0][1] = 2
        elif coords[0][0]-25<x_coord<coords[0][0]+25 and coords[0][1]-25<y_coord<coords[0][1]+25:
            position[0][0] = 2


    #red
    temp = cv2.addWeighted(red, 0.4, green, -0.2, 50)
    exr = cv2.addWeighted(temp, 1, blue, -0.2, 0)

    # Find places with an exr value over 80 (it suits for this video, but probably not yours)
    retval, thrsholdeded = cv2.threshold(exr, 60, 255, cv2.THRESH_BINARY)

    # show the thresholded image
    #cv2.imshow('exr', thrsholdeded)


    imcleaned = morphology.remove_small_objects(thrsholdeded>25, min_size=552, connectivity=2)
    _, contours, hierarchy = cv2.findContours((imcleaned*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow('cleaned', (imcleaned*255).astype(np.uint8))
    for cnt in contours:
   
        area = cv2.contourArea(cnt)
        circumference = cv2.arcLength(cnt, True)
        if 1 < 2:
        #</alter these lines>
            # Calculate the centre of mass
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            red_brickCoords.append((cx,cy))

    for coord in red_brickCoords:
    
        x_coord, y_coord = coord
        if coords[8][0] -25<x_coord<coords[8][0]+25 and coords[8][1]-25<y_coord<coords[8][1]+25:
            position[2][2] = 1
        elif coords[7][0]-25<x_coord<coords[7][0]+25 and coords[7][1]-25<y_coord<coords[7][1]+25:
            position[2][1] = 1
        elif coords[6][0]-25<x_coord<coords[6][0]+25 and coords[6][1]-25<y_coord<coords[6][1]+25:
            position[2][0] = 1
        elif coords[5][0]-25<x_coord<coords[5][0]+25 and coords[5][1]-25<y_coord<coords[5][1]+25:
            position[1][2] = 1
        elif coords[4][0]-25<x_coord<coords[4][0]+25 and coords[4][1]-25<y_coord<coords[4][1]+25:
            position[1][1] = 1
        elif coords[3][0]-25<x_coord<coords[3][0]+25 and coords[3][1]-25<y_coord<coords[3][1]+25:
            position[1][0] = 1
        elif coords[2][0]-25<x_coord<coords[2][0]+25 and coords[2][1]-25<y_coord<coords[2][1]+25:
            position[0][2] = 1
        elif coords[1][0]-25<x_coord<coords[1][0]+25 and coords[1][1]-25<y_coord<coords[1][1]+25:
            position[0][1] = 1
        elif coords[0][0]-25<x_coord<coords[0][0]+25 and coords[0][1]-25<y_coord<coords[0][1]+25:
            position[0][0] = 1

    #return position


    #print(contours)
    #print(hierarchy)
    for array in position:
        print(array)

    print('Red brick position', red_brickCoords)
    print('Blue brick position', blue_brickCoords)
    
    #imcontours = cv2.drawContours(frame, [contours[1]], -1, (0,255,0), 3)
    #image = cv2.circle(frame, red_brickCoords[1], radius=0, color=(0, 0, 255), thickness=10)
    while(done):
        #cv2.imshow('it works',imcontours)
        #cv2.imshow("image",image)
        # Display the original image and the results
        cv2.imshow('Original Image', frame)
        
        key = cv2.waitKey(1)
    