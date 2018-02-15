#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

def find_homography(frame):    
    markerBottomLeftCorners = [[0, -0.6], [0.6, -0.6], [0, -0.3], [0.6, 0], [0.6, 0.3], [-0.3, 0]]
    markerIds = np.array([27, 18, 5, 12, 43, 42])
    markerWidth = 0.035

    # create the aruco board given marker positions and ids
    objPoints = []
    for i in range(0, len(markerBottomLeftCorners)):
        [x, y] = markerBottomLeftCorners[i]
        #corners = np.array([
        #    x, y, 0,
        #    x, y + markerWidth, 0,
        #    x + markerWidth, y + markerWidth, 0,
        #    x + markerWidth, y, 0,
        #])
        corners = np.array([
            x - markerWidth / 2, y - markerWidth / 2, 0,
            x - markerWidth / 2, y + markerWidth / 2, 0,
            x + markerWidth / 2, y + markerWidth / 2, 0,
            x + markerWidth / 2, y - markerWidth / 2, 0,
        ])
        objPoints.append(corners)
    objPoints = np.array(objPoints, ('float32'))
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    board = aruco.Board_create(objPoints, aruco_dict, markerIds)

    # use default aruco detection parameters
    parameters = aruco.DetectorParameters_create()

    # TODO: convert to grayscale if necessary
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # threshold the image
    gray_thresh = frame.copy()
    cv2.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                          thresholdType=cv2.THRESH_BINARY, blockSize=5, C=0, dst=gray_thresh)

    # detect markers and then refine based on the known board positions
    corners, ids, rejectedCorners = aruco.detectMarkers(gray_thresh, aruco_dict, parameters=parameters)
    # print("corners: ", corners)
    corners, ids, rejectedCorners, _ = aruco.refineDetectedMarkers(gray_thresh, board, corners, ids, rejectedCorners)
    # print("refined corners: ", corners)

    # find the homography mapping world to image coordinates
    boardPoints, imgPoints = aruco.getBoardObjectAndImagePoints(board, corners, ids)
    H, _ = cv2.findHomography(boardPoints, imgPoints, cv2.LMEDS)

    return H, corners
