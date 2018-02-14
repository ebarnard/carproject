import cv2
import cv2.aruco as aruco
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import csv


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

cap = cv2.VideoCapture("../video/car_drive_demo.avi")

ok, frame = cap.read()
frame = cv2.imread("../video/aruco_4x4_50_test_board.png", cv2.IMREAD_GRAYSCALE)

# for i in range(0, 1000):
#     cap.read()

first_run = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.imread("../video/aruco_4x4_50_test_board.png", cv2.IMREAD_GRAYSCALE)
    # print(frame.shape) #480x640
    # Our operations on the frame come here
    # frame = frame[400:880, 300:940] #480x640
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_thresh = frame.copy()
    cv2.adaptiveThreshold(frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=0,
                          dst=gray_thresh)
    # MDetector.setDictionary("ARUCO_MIP_36h12")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    markerBottomLeftCorners = [[0, -0.6], [0.6, -0.6], [0, -0.3], [0.6, 0], [0.6, 0.3], [-0.3, 0]]
    markerWidth = 0.035
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

    objPoints = np.array(objPoints ,('float32'))
    ids = np.array([27, 18, 5, 12, 43, 42])
    board = aruco.Board_create(objPoints, aruco_dict, ids)

    parameters = aruco.DetectorParameters_create()
    # lists of ids and the corners belonging to each id
    corners, ids, rejectedCorners = aruco.detectMarkers(gray_thresh, aruco_dict, parameters=parameters)
    # print("corners: ", corners)

    corners, ids, rejectedCorners, _ = aruco.refineDetectedMarkers(gray_thresh, board, corners, ids, rejectedCorners)
    # print("refined corners: ", corners)

    boardPoints, imgPoints = aruco.getBoardObjectAndImagePoints(board, corners, ids)
    H, _ = cv2.findHomography(boardPoints, imgPoints, cv2.LMEDS)
    H_inv = linalg.inv(H)
    H_inv = H_inv / H_inv[2, 2]

    one = np.array([0, 0.3, 1])
    track_centreline = np.genfromtxt('../track_model_generator/office_desk_track_2500.csv', dtype=None, delimiter=',', skip_header=1)
    gradient = (np.roll(track_centreline, -3) - np.roll(track_centreline, 3)) / 2
    leng = np.hypot(gradient[:, 0], gradient[:, 1])
    gradient[:, 0] /= leng
    gradient[:, 1] /= leng
    normal = np.array([-gradient[:, 1], gradient[:, 0]])
    track_inner = np.array([track_centreline[:, 0] + normal[0, :] * (track_centreline[:, 2]/2),
                            track_centreline[:, 1] + normal[1, :] * (track_centreline[:, 2]/2)])
    track_outer = np.array([track_centreline[:, 0] - normal[0, :] * (track_centreline[:, 2]/2),
                            track_centreline[:, 1] - normal[1, :] * (track_centreline[:, 2]/2)])

    track_inner = H.dot(np.r_[track_inner, [np.ones(2500)]])
    track_outer = H.dot(np.r_[track_outer, [np.ones(2500)]])

    plt.plot(track_centreline[:, 0], track_centreline[:, 1], "r")
    plt.plot(track_inner[0, :], track_inner[1, :], "g")
    plt.plot(track_outer[0, :], track_outer[1, :], "b")
    plt.axis("equal")

    img = np.zeros((1024, 1280), np.uint8)
    track_outer_reshape = track_outer.transpose()

    cv2.fillPoly(img, np.array([track_outer[0:2, :].transpose(), track_inner[0:2, :].transpose()], np.int32), 255)

    v = H.dot(one)
    print("world origin", v / v[2])

    v_world = H_inv.dot(v/v[2])
    print("world origin in meters", v_world/v_world[2])

    # It's working.
    # my problem was that the cellphone put black all around it. The algorithm
    # depends very much upon finding rectangular black blobs


    # aruco.calibrateCameraAruco()

    frame = aruco.drawDetectedMarkers(frame, corners, borderColor=127)

    # print(rejectedImgPoints)
    # Display the resulting frame
    frame = cv2.add(frame, img)

    cv2.imshow('frame', cv2.resize(frame, (int(1280 * 0.5), int(1024 * 0.5))))


    plt.show()

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
