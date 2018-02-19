#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np

def find_homography(frame):    
    markerBottomLeftCorners = [[-0.305, -0.490], [-0.902, 0.450], [0.848, 0.440], [0.848, -0.498], [0.298, -0.178],
                               [-0.162, 0.246]]
    markerIds = np.array([8, 10, 11, 13, 16, 14])
    markerWidth = 0.060

    # create the aruco board given marker positions and ids
    objPoints = []
    for i in range(0, len(markerBottomLeftCorners)):
        [x, y] = markerBottomLeftCorners[i]
        corners = np.array([
            x, y, 0,
            x, y + markerWidth, 0,
            x + markerWidth, y + markerWidth, 0,
            x + markerWidth, y, 0,
        ])
        objPoints.append(corners)
    objPoints = np.array(objPoints, ('float32'))
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    board = aruco.Board_create(objPoints, aruco_dict, markerIds)

    # use default aruco detection parameters
    parameters = aruco.DetectorParameters_create()

    # detect markers and then refine based on the known board positions
    corners, ids, rejectedCorners = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    # print("corners: ", corners)
    corners, ids, rejectedCorners, _ = aruco.refineDetectedMarkers(frame, board, corners, ids, rejectedCorners)
    # print("refined corners: ", corners)

    # find the homography mapping world to image coordinates
    boardPoints, imgPoints = aruco.getBoardObjectAndImagePoints(board, corners, ids)
    H, _ = cv2.findHomography(boardPoints, imgPoints, cv2.LMEDS)

    return H, corners
