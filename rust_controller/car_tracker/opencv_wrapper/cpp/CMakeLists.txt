CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(opencv_wrapper_project)
FIND_PACKAGE(OpenCV REQUIRED)

# Create static library
ADD_LIBRARY(opencv_wrapper SHARED opencv_wrapper.cpp)

# Link with OpenCV
target_link_libraries(opencv_wrapper ${OpenCV_LIBS})

# Install to default install dir
install(TARGETS opencv_wrapper DESTINATION ./)
