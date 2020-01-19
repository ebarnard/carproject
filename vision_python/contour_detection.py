from PIL import Image
from timeit import default_timer as timer
import cv2
import numpy as np
import math
import PyCapture2
from PyCapture2 import Camera, BusManager

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
video = cv2.VideoCapture("../video/car_drive_demo.avi")

ok, frame = video.read()

for i in range(0, 150):
    video.read()

# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frame, False)

i = 0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
    time_start = timer()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 35, 255, 0)

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
    flooded = thresh.copy()
    cv2.floodFill(flooded, mask, (bbox[0], bbox[1]), 255)

    image_use = 255 - (flooded - thresh)
    kernel = np.ones((10, 10), np.uint8)
    image_use = cv2.morphologyEx(image_use, cv2.MORPH_CLOSE, kernel)

    if i == 0:
        image_average = image_use
        i += 1

    cv2.addWeighted(image_average, 0.95, image_use, 0.05, -1, image_average)
    ret, track = cv2.threshold(image_average, 240, 255, 0)

    image_use = image_use*(track < 240)

    im2, contours, hierarchy = cv2.findContours(image_use, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centres_of_mass = []
    # loop over the contours
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        elif M["m00"] > 5000:
            continue
        elif M["m00"] < 1000:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        print("Coordinates of marker", len(centres_of_mass) + 1, ":")
        print(cX, ",", cY)

        # Adds the coordinates of the centre of mass to an array
        centres_of_mass.append([cX, cY])

        # draw the contour and center of the shape on the image
        cv2.drawContours(im2, [c], -1, (120, 120, 120), 3)
        cv2.circle(im2, (cX, cY), 3, (50, 50, 50), -1)
        cv2.putText(im2, "center", (cX - 30, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # show the image
        cv2.imshow("Image", im2)
        cv2.waitKey(1)
    time_delay = timer() - time_start
    print(time_delay)

# new_image = cv2.imread('single_square_test_result_48.png')
# imgray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
# new_ret, new_thresh = cv2.threshold(imgray, 255, 255, 0)
# new_im2, new_contours, new_hierarchy = cv2.findContours(new_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in centres_of_mass:
#     cv2.circle(new_im2, (c[0], c[1]), 5, (255, 255, 255), -1)
#     cv2.putText(new_im2, "center", (c[0] - 10, c[1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
# # show centre of mass points
# cv2.imshow("Image", new_im2)
# cv2.waitKey(0)
