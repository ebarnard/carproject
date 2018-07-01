#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

extern "C" void draw_car_position(
    unsigned char* data,
    unsigned int width,
    unsigned int height,
    double x,
    double y,
    double heading
) {
    Mat img = Mat(height, width, CV_8U, (void*)data);

    rectangle(img, Rect(x - 5, y - 5, 10, 10), Scalar(127), 5);

    int heading_x = cos(heading) * 55.0 + x;
    int heading_y = sin(heading) * 55.0 + y;
    line(img, Point(x, y), Point(heading_x, heading_y), Scalar(127), 3);
}

extern "C" void show_greyscale_image(
    const unsigned char* const data,
    unsigned int width,
    unsigned int height,
    unsigned int delay
) {
    Mat data_mat = Mat(height, width, CV_8U, (void*)data);
    imshow("win", data_mat);
    waitKey(delay);
}
