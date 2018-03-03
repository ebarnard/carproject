cmake_minimum_required(VERSION 2.8)
project(carproject)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(contour_detection contour_detection_grayscale_functions.cpp)
target_link_libraries(contour_detection ${OpenCV_LIBS})
