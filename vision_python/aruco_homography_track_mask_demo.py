import cv2
import cv2.aruco as aruco
import numpy as np

import aruco_markers
import track

frame = cv2.imread("../video/aruco_4x4_50_test_board.png", cv2.IMREAD_GRAYSCALE)
[h, w] = frame.shape

H, corners = aruco_markers.find_homography(frame)
track_mask = track.create_mask('../track_model_generator/office_desk_track_2500.csv', H, w, h)

# draw detected markers and track mask
frame = aruco.drawDetectedMarkers(frame, corners, borderColor=127)
frame = cv2.add(frame, track_mask)

cv2.imshow('frame', cv2.resize(frame, (int(w * 0.5), int(h * 0.5))))
cv2.waitKey(0)
cv2.destroyAllWindows()
