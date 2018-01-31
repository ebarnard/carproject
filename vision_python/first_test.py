from PIL import Image
import cv2
import numpy as np

# Open image, convert it to gray scale image and then use a threshold to convert that into black and white image
colour = Image.open("single_square_test.png")
gray = colour.convert('L')
black_white = gray.point(lambda x: 0 if x < 48 else 255, '1')
black_white.save("single_square_test_result_48.png")

# Read image
image = cv2.imread("single_square_test_result_48.png", cv2.IMREAD_GRAYSCALE)

# # objective from week 5
# cv2.imshow("image", image)
# cv2.waitKey(0)


# # Set up the SimpleBlobdetector with default parameters.
# params = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
# params.minThreshold = 0
# params.maxThreshold = 256
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 3
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.5
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.5
#
# detector = cv2.SimpleBlobDetector_create(params)


# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
reverseimage = 255 - image
keypoints = detector.detect(reverseimage)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)