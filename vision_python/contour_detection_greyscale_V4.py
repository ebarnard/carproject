from PIL import Image
from timeit import default_timer as timer
import cv2
import numpy as np
import math
import PyCapture2
from PyCapture2 import Camera, BusManager
import track
import aruco_markers

#
# def printBuildInfo():
#     lib_ver = PyCapture2.getLibraryVersion()
#     print("PyCapture2 library version: ", lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3])
#     print()
#
#
# def printCameraInfo(cam):
#     cam_info = cam.getCameraInfo()
#     print("\n*** CAMERA INFORMATION ***\n")
#     print("Serial number - ", cam_info.serialNumber)
#     print("Camera model - ", cam_info.modelName)
#     print("Camera vendor - ", cam_info.vendorName)
#     print("Sensor - ", cam_info.sensorInfo)
#     print("Resolution - ", cam_info.sensorResolution)
#     print("Firmware version - ", cam_info.firmwareVersion)
#     print("Firmware build time - ", cam_info.firmwareBuildTime)
#     print()
#
#
# def saveAviHelper(cam, fileFormat, fileName, frameRate):
#     num_images = 1000
#     avi = PyCapture2.AVIRecorder()
#     for i in range(num_images):
#         try:
#             image = cam.retrieveBuffer()
#         except PyCapture2.Fc2error as fc2Err:
#             print("Error retrieving buffer : ", fc2Err)
#             continue
#         print("Grabbed image {}".format(i))
#         if (i == 0):
#             # if fileFormat == "AVI":
#             #     avi.AVIOpen(fileName, frameRate)
#             if fileFormat == "MJPG":
#                 avi.MJPGOpen(fileName, frameRate, 75)
#             # elif fileFormat == "H264":
#             #     avi.H264Open(fileName, frameRate, image.getCols(), image.getRows(), 1000000)
#             else:
#                 print("Specified format is not available.")
#                 return
#         avi.append(image)
#         print("Appended image {}...".format(i))
#     print("Appended {} images to {} file: {}...".format(num_images, fileFormat, fileName))
#     avi.close()
#
#
# def enableEmbeddedTimeStamp(cam, enableTimeStamp):
#     embeddedInfo = cam.getEmbeddedImageInfo()
#     if embeddedInfo.available.timestamp:
#         cam.setEmbeddedImageInfo(timestamp=enableTimeStamp)
#         if (enableTimeStamp):
#             print("\nTimeStamp is enabled.\n")
#         else:
#             print("\nTimeStamp is disabled.\n")
#
#
# def grabImages(cam, numImagesToGrab):
#     prevts = None
#     for i in range(numImagesToGrab):
#         try:
#             image = cam.retrieveBuffer()
#         except PyCapture2.Fc2error as fc2Err:
#             print("Error retrieving buffer : ", fc2Err)
#             continue
#
#         ts = image.getTimeStamp()
#         if (prevts):
#             diff = (ts.cycleSeconds - prevts.cycleSeconds) * 8000 + (ts.cycleCount - prevts.cycleCount)
#             print("Timestamp [", ts.cycleSeconds, ts.cycleCount, "] -", diff)
#         prevts = ts
#
#     im = np.array(image.getData()).reshape((image.getRows(), image.getCols()))
#
#     cv2.imshow("Image", im)
#     cv2.waitKey(0)
#
#     ret, thresh = cv2.threshold(im, 48, 255, 0)
#     cv2.imshow("Image", thresh)
#     cv2.waitKey(0)
#
#
# # create instances of the object BusManager and check number of cameras
# bus = PyCapture2.BusManager()
# numCams = bus.getNumOfCameras()
# print("Number of cameras detected: ", numCams)
# if not numCams:
#     print("Insufficient number of cameras. Exiting...")
#     exit()
#
# # Select camera on 0th index
# camera_serial_number = 17115008
# camera_guit = bus.getCameraFromSerialNumber(camera_serial_number)
# cam = PyCapture2.Camera()
# cam.connect(bus.getCameraFromIndex(0))
#
# # Print camera details
# printCameraInfo(cam)
#
# # Set format to mono 8
# fmt7info, supported = cam.getFormat7Info(0)
#
# # Check whether pixel format mono8 is supported
# if PyCapture2.PIXEL_FORMAT.MONO8 & fmt7info.pixelFormatBitField == 0:
#     print("Pixel format is not supported\n")
#     exit()
#
# # Configure camera format7 settings
# fmt7imgSet = PyCapture2.Format7ImageSettings(0, 0, 0, fmt7info.maxWidth, fmt7info.maxHeight, PyCapture2.PIXEL_FORMAT.MONO8)
# fmt7pktInf, isValid = cam.validateFormat7Settings(fmt7imgSet)
# if not isValid:
#     print("Format7 settings are not valid!")
#     exit()
# cam.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
#
# # Enable camera embedded timestamp
# enableEmbeddedTimeStamp(cam, True)
#
# print("Starting image capture...")
# cam.startCapture()
# grabImages(cam, 100)
# cam.stopCapture()
#
# # Disable camera embedded timestamp
# enableEmbeddedTimeStamp(cam, False)
#
# cam.disconnect()
#
# input("Done! Press Enter to exit...\n")

# Read video
video = cv2.VideoCapture("../video/2_cars_with_markers_drive_bump_demo.avi")

ok, frame = video.read()

frame_track = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame = cv2.imread("../video/aruco_4x4_50_new_board.png", cv2.IMREAD_GRAYSCALE)

[h, w] = frame_track.shape
H, corners = aruco_markers.find_homography(frame)
track_mask = track.create_mask('../track_model_generator/office_desk_track_2500.csv', H, w, h)

# # show the image
# cv2.imshow('track mask', cv2.resize(track_mask, (int(w * 0.7), int(h * 0.7))))
# cv2.waitKey(1)

# for i in range(0, 250):
#     video.read()

cars = {}
cars['car 1'] = "empty"
cars['car 2'] = "empty"
first_run = True
car_1_position_updated = False
car_2_position_updated = False
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

    # if first_run:
    #     image_average = thresh
    #     first_run = False
    #
    # cv2.addWeighted(image_average, 0.95, thresh, 0.05, -1, image_average)
    # ret, track = cv2.threshold(image_average, 240, 255, 0)
    #
    # thresh = thresh * (track < 240)
    #
    # thresh = cv2.medianBlur(thresh, 5)

    # takes around 1ms
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    i = 0
    car_1_position_updated = False
    car_2_position_updated = False
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

        cX, cY = np.int0(rect[0])

        if "car 1" and "car 2" in cars.keys():
            for key in cars:
                if cars[key] is not "empty":
                    car_rect = cv2.minAreaRect(cars[key])
                    car_X, car_Y = np.int0(car_rect[0])
                    distance = math.sqrt((car_X - cX)*(car_X - cX) + (car_Y - cY)*(car_Y - cY))
                    # draw the contour and center of the shape on the image
                    if distance < 50:
                        if key == "car 1":
                            car_1_position_updated = True
                        elif key == "car 2":
                            car_2_position_updated = True
                        cars[key] = c
                elif cars[key] is "empty":
                    if key == "car 1":
                        if cars['car 2'] is not "empty":
                            check_rect = cv2.minAreaRect(cars['car 2'])
                            check_X, check_Y = np.int0(check_rect[0])
                            if check_X == cX and check_Y == cY:
                                continue
                            else:
                                pass
                        else:
                            pass
                    if key == "car 2":
                        if cars['car 1'] is not "empty":
                            check_rect = cv2.minAreaRect(cars['car 1'])
                            check_X, check_Y = np.int0(check_rect[0])
                            if check_X == cX and check_Y == cY:
                                continue
                            else:
                                pass
                        else:
                            pass

                    cars.pop(key)
                    cars[key] = c
                    if key == "car 1":
                        car_1_position_updated = True
                    elif key == "car 2":
                        car_2_position_updated = True

            i += 1
            if i == 1:
                for key in cars:
                    if key is "car 1" and cars[key] is not "empty":
                        car_rect = cv2.minAreaRect(cars[key])
                        car_X, car_Y = np.int0(car_rect[0])
                        rect_corners = np.int0(cv2.boxPoints(car_rect))
                        print("Coordinates of ", key, " :")
                        print(car_X, ",", car_Y, ",", car_rect[2])
                        cv2.drawContours(im2, [rect_corners], -1, (150, 0, 0), 5)
                        cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
                        cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif key is "car 2" and cars[key] is not "empty":
                        car_rect = cv2.minAreaRect(cars[key])
                        car_X, car_Y = np.int0(car_rect[0])
                        rect_corners = np.int0(cv2.boxPoints(car_rect))
                        print("Coordinates of ", key, " :")
                        print(car_X, ",", car_Y, ",", car_rect[2])
                        cv2.drawContours(im2, [rect_corners], -1, (50, 0, 0), 5)
                        cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
                        cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if car_1_position_updated is False:
        print("car 1 position is not detected!")
        cars.pop('car 1')
        cars['car 1'] = "empty"
    if car_2_position_updated is False:
        print("car 2 position is not detected!")
        cars.pop('car 2')
        cars['car 2'] = "empty"

    time_delay = timer() - time_start
    print(time_delay)
    # show the image
    cv2.imshow('Image', cv2.resize(im2, (int(w * 0.7), int(h * 0.7))))
    cv2.waitKey(1)
