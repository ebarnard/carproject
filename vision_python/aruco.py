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

cap = cv2.VideoCapture("../video/aruco_sheet.avi")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_thresh = gray.copy()
    cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=0, thresholdType=0, blockSize=9, C=15,
                          dst=gray_thresh)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

    parameters = aruco.DetectorParameters_create()

    markerImage = np.ones((200, 200), np.uint8)
    cv2.drawMarker(aruco_dict, 23, 200, markerImage, 1)
    print("parameters: ", parameters)

    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
    # lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_thresh, aruco_dict, parameters=parameters)
    print("corners: ", corners)

    # It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs

    gray_thresh = aruco.drawDetectedMarkers(gray_thresh, corners)

    # print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame', gray_thresh)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
