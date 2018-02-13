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

    gray = cv2.imread("../video/aruco_4x4_50_test_board.png", cv2.IMREAD_GRAYSCALE)
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    # frame = frame[400:880, 300:940] #480x640
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_thresh = gray.copy()
    cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=0,
                          dst=gray_thresh)
    # MDetector.setDictionary("ARUCO_MIP_36h12")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    markerBottomLeftCorners = [[0, 0], [0.6, 0], [0, 0.3], [0.6, 0.6], [0.6, 0.9], [-0.3, 0.6]]
    markerWidth = 0.035
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

    objPoints = np.array(objPoints ,('float32'))
    ids = np.array([27, 18, 5, 12, 43, 42])
    board = aruco.Board_create(objPoints, aruco_dict, ids)

    parameters = aruco.DetectorParameters_create()
    # lists of ids and the corners belonging to each id
    corners, ids, rejectedCorners = aruco.detectMarkers(gray_thresh, aruco_dict, parameters=parameters)
    print("corners: ", corners)

    corners, ids, rejectedCorners, _ = aruco.refineDetectedMarkers(gray_thresh, board, corners, ids, rejectedCorners)
    print("refined corners: ", corners)

    # Calibration requires a weird input format
    cornersConcatenated = []
    idsConcatenated = []
    markerCounterPerFrame = []

    for i in range(0, len(ids)):
        markerCounterPerFrame.append(len(corners[i]))
        for j in range(0, len(corners[i])):
            cornersConcatenated.append(corners[i][j])
            idsConcatenated.append(ids[i][j])

    cornersConcatenated = np.array(cornersConcatenated)
    idsConcatenated = np.array(idsConcatenated)
    markerCounterPerFrame = np.array(markerCounterPerFrame)

    retval, cameraMatrix, distCoeffs, _, _ = aruco.calibrateCameraAruco(cornersConcatenated, idsConcatenated, markerCounterPerFrame, board, gray.shape, None, None)

    ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)
    R, _ = cv2.Rodrigues(rvec)

    P = np.zeros((3, 3))
    P[:, 0:2] = R[:, 0:2]
    P[:, 2] = tvec[:, 0]
    P /= P[2, 2]
    H = cameraMatrix.dot(P)

    one = np.array([0, 0, 1])
    v = H.dot(one)
    print("world origin", v / v[2])

    print("camera matrix", cameraMatrix)
    print("rot", R)
    print("tr", tvec)

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
