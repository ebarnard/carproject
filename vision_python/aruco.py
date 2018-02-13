import cv2
import cv2.aruco as aruco
import numpy as np

# '''
#     drawMarker(...)
#         drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
# '''
#
# aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
# print(aruco_dict)
# # second parameter is id number
# # last parameter is total image size
# img = aruco.drawMarker(aruco_dict, 2, 700)
# cv2.imwrite("test_marker.jpg", img)
#
# cv2.imshow('frame', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture("../video/aruco_sheet_4x4_50_small.avi")

# for i in range(0, 1000):
#     cap.read()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    # frame = frame[400:880, 300:940] #480x640
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_thresh = gray.copy()
    cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=0,
                          dst=gray_thresh)
    # MDetector.setDictionary("ARUCO_MIP_36h12")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    parameters = aruco.DetectorParameters_create()
    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_thresh, aruco_dict, parameters=parameters)
    print("corners: ", corners)

    # It's working.
    # my problem was that the cellphone put black all around it. The algorithm
    # depends very much upon finding rectangular black blobs




    # aruco.calibrateCameraAruco()
    gray = aruco.drawDetectedMarkers(gray, corners, borderColor=127)

    # print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
