from PIL import Image
from timeit import default_timer as timer
import cv2
import numpy as np
import math
# import PyCapture2
# from PyCapture2 import Camera, BusManager
import track
import aruco_markers

# Read video
video = cv2.VideoCapture("../video/2_cars_with_markers_drive_bump_demo.avi")

ok, frame = video.read()

frame_track = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame = cv2.imread("../video/aruco_4x4_50_new_board.png", cv2.IMREAD_GRAYSCALE)

[h, w] = frame_track.shape
H, corners = aruco_markers.find_homography(frame)
track_mask = track.create_mask('../track_model_generator/office_desk_track_2500.csv', H, w, h)

contour = None
empty = True
name_1 = 'car 1'
name_2 = 'car 2'
cars_ = [[empty, contour, name_1], [empty, contour, name_2]]
car_1_position_updated = False
car_2_position_updated = False
car_detected = False
while True:
    time_start = timer()
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    # takes around 0.8ms
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # takes around 3ms
    ret, thresh = cv2.threshold(imgray, 60, 255, 0)
    thresh = thresh * (track_mask == 255)

    # takes around 3ms
    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # takes around 1ms
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    i = 0
    car_1_position_updated = False
    car_2_position_updated = False
    car_detected = False
    for c in contours:
        # Looks like: ((centre_x, centre_y), (width, height), angle)
        rect = cv2.minAreaRect(c)

        length = max(rect[1])
        width = min(rect[1])

        min_length = 50
        max_length = 120
        min_width = 40
        max_width = 70

        if length > min_length and length < max_length and width > min_width and width < max_width:
            pass
        else:
            continue

        car_detected = True

        cX, cY = np.int0(rect[0])

        for car in cars_:
            if not car[0]:
                car_rect = cv2.minAreaRect(car[1])
                car_X, car_Y = np.int0(car_rect[0])
                distance = math.sqrt((car_X - cX) * (car_X - cX) + (car_Y - cY) * (car_Y - cY))
                if distance < 50:
                    if car[2] == "car 1":
                        car_1_position_updated = True
                    elif car[2] == "car 2":
                        car_2_position_updated = True
                    car[1] = c
                    car[0] = False
            elif car[0]:
                if car[2] == "car 1":
                    if not cars_[1][0]:
                        check_rect = cv2.minAreaRect(cars_[1][1])
                        check_X, check_Y = np.int0(check_rect[0])
                        if check_X == cX and check_Y == cY:
                            continue
                        else:
                            pass
                    else:
                        pass
                if car[2] == "car 2":
                    if not cars_[0][0]:
                        check_rect = cv2.minAreaRect(cars_[0][1])
                        check_X, check_Y = np.int0(check_rect[0])
                        if check_X == cX and check_Y == cY:
                            continue
                        else:
                            pass
                    else:
                        pass
                car[1] = c
                car[0] = False
                if car[2] == "car 1":
                    car_1_position_updated = True
                elif car[2] == "car 2":
                    car_2_position_updated = True

        i += 1
        if i == 1:
            for car in cars_:
                if car[2] == "car 1" and not car[0]:
                    car_rect = cv2.minAreaRect(car[1])
                    car_X, car_Y = np.int0(car_rect[0])
                    rect_corners = np.int0(cv2.boxPoints(car_rect))
                    print("Coordinates of ", car[2], " :")
                    print(car_X, ",", car_Y, ",", car_rect[2])
                    cv2.drawContours(im2, [rect_corners], -1, (150, 0, 0), 5)
                    cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
                    cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                elif car[2] == "car 2" and not car[0]:
                    car_rect = cv2.minAreaRect(car[1])
                    car_X, car_Y = np.int0(car_rect[0])
                    rect_corners = np.int0(cv2.boxPoints(car_rect))
                    print("Coordinates of ", car[2], " :")
                    print(car_X, ",", car_Y, ",", car_rect[2])
                    cv2.drawContours(im2, [rect_corners], -1, (50, 0, 0), 5)
                    cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
                    cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    if car_1_position_updated is False:
        print("car 1 position is not detected!")
        cars_[0][0] = True
    if car_2_position_updated is False:
        print("car 2 position is not detected!")
        cars_[1][0] = True

    time_delay = timer() - time_start
    print(time_delay)
    # show the image
    cv2.imshow('Image', cv2.resize(im2, (int(w * 0.7), int(h * 0.7))))
    cv2.waitKey(1)
