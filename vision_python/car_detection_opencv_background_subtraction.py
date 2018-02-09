from PIL import Image
from timeit import default_timer as timer
import cv2
import numpy as np
import math
import PyCapture2
from PyCapture2 import Camera, BusManager


# Read video
video = cv2.VideoCapture("../video/car_drive_demo.avi")

bgrm = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)

for i in range(0, 150):
    video.read()

for i in range(0, 150):
    ok, frame = video.read()
    if not ok:
        break

    bgrm.apply(frame)

bgim = bgrm.getBackgroundImage()

# show the image
cv2.imshow("Image", bgim)
cv2.waitKey(0)

i = 0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    time_start = timer()

    fgmask = bgrm.apply(frame, learningRate=0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    #cv2.imshow('fg', fgmask)
    #cv2.waitKey(0)

    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    centres_of_mass = []
    # loop over the contours
    for c in contours:
        # Looks like: ((centre_x, centre_y), (width, height), angle)
        rect = cv2.minAreaRect(c)

        length = max(rect[1])
        width = min(rect[1])

        min_length = 65
        max_length = 100
        min_width = 30
        max_width = 50

        if length > min_length and length < max_length and width > min_width and width < max_width:
            pass
        else:
            continue

        cX, cY = np.int0(rect[0])

        print("Coordinates of marker", len(centres_of_mass) + 1, ":")
        print(cX, ",", cY)

        # Adds the coordinates of the centre of mass to an array
        centres_of_mass.append([cX, cY])

        # draw the contour and center of the shape on the image
        rect_corners = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(im2, [rect_corners], -1, (120, 120, 120), 3)
        cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
        cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the image
    cv2.imshow("Image", im2)
    cv2.waitKey(1)

    time_delay = timer() - time_start
    print(time_delay)