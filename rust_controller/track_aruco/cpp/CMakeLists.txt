CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(track_aruco_project)
FIND_PACKAGE(OpenCV REQUIRED)

# Create static library
ADD_LIBRARY(track_aruco SHARED track_aruco.cpp)

# Link with OpenCV
target_link_libraries(track_aruco ${OpenCV_LIBS})

# Install to default install dir
install(TARGETS track_aruco DESTINATION ./)
