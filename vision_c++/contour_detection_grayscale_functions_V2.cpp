#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

using namespace cv;
using namespace std;

struct CarPosition {
    double x;
    double y;
    double heading;
};

struct Car {
    bool empty = true;
    bool position_updated = false;
    Mat contour;
    CarPosition position;
};

vector<CarPosition> position_update (Car* cars, int number_of_cars) {
    vector<CarPosition> positions;
    for (int k = 0; k < number_of_cars; k++) {
        if (!cars[k].empty) {
            RotatedRect car_rect = minAreaRect(cars[k].contour);
            double car_X = car_rect.center.x;
            double car_Y = car_rect.center.y;
            cars[k].position.x = car_X;
            cars[k].position.y = car_Y;
            cars[k].position.heading = car_rect.angle;
            positions.push_back(cars[k].position);
        };
    };
    return positions;
};

void car_contour_filtering (Car* cars, int number_of_cars, Mat* contour) {
    RotatedRect rect = minAreaRect(*contour);

    int length = max(int(rect.size.height), int(rect.size.width));
    int width = min(int(rect.size.height), int(rect.size.width));

    int min_length = 50;
    int max_length = 120;
    int min_width = 40;
    int max_width = 70;

    if (length < min_length || length > max_length || width < min_width || width > max_width) {
        return;
    };
    double cX = rect.center.x;
    double cY = rect.center.y;
    for (int k = 0; k < number_of_cars; k++) {
        if (!cars[k].empty) {
            RotatedRect car_rect = minAreaRect(cars[k].contour);
            double car_X = car_rect.center.x;
            double car_Y = car_rect.center.y;
            double distance = sqrt((car_X - cX) * (car_X - cX) + (car_Y - cY) * (car_Y - cY));
            if (distance < 50) {
                cars[k].position_updated = true;
                cars[k].contour = *contour;
                cars[k].empty = false;
            };
        } else if (cars[k].empty) {
            if (k == 0) {
                if (!cars[1].empty) {
                    RotatedRect check_rect = minAreaRect(cars[1].contour);
                    double check_X = check_rect.center.x;
                    double check_Y = check_rect.center.y;
                    if (int(check_X) == int(cX) && int(check_Y) == int(cY)) {
                        continue;
                    };
                };
                cars[k].position_updated = true;
            } else if (k == 1) {
                if (!cars[0].empty) {
                    RotatedRect check_rect = minAreaRect(cars[0].contour);
                    double check_X = check_rect.center.x;
                    double check_Y = check_rect.center.y;
                    if (int(check_X) == int(cX) && int(check_Y) == int(cY)) {
                        continue;
                    };
                };
                cars[k].position_updated = true;
            };
            cars[k].contour = *contour;
            cars[k].empty = false;
        };
    };
};

Mat frame_filtering (Mat& image, Mat track_mask, Mat_<int> kernel) {
    threshold(image, image, 60, 255, 0);

    Mat imgray_2(image.size(), CV_8UC1);
    image.copyTo(imgray_2, track_mask);
    imgray_2.copyTo(image);

    morphologyEx(image, image, MORPH_OPEN, kernel);
    morphologyEx(image, image, MORPH_CLOSE, kernel);

    return image;
};

class Tracker {
public:
    Car cars[2];
    Mat imgray;
    Mat track_mask = imread("..\\track_mask.png", 0);
    Mat_<int> kernel;
    vector<Mat> contours;

    Tracker (Mat frame) {
        Mat imgray(frame.size(), CV_8UC1);
        for(int a = 0; a < kernel.rows; a++) {
            for (int b = 0; b < kernel.cols; b++) {
                kernel(a, b) = 1;
            };
        };

    };

    vector<CarPosition> car_position_detecting (Mat* frame) {
        vector<CarPosition> positions;
        cvtColor(*frame, imgray, CV_BGR2GRAY);

        imgray = frame_filtering(imgray, track_mask, kernel);

        findContours(imgray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        cars[0].position_updated = false;
        cars[1].position_updated = false;
        for(int j = 0; j < contours.size(); j += 1) {
            car_contour_filtering(cars, 2, &contours[j]);
            if (j == (contours.size()-1)) {
                positions = position_update(cars, 2);
            };
        };

        if(!cars[0].position_updated) {
            cout<<"car 1 position is not detected!" << endl;
            cars[0].empty = true;
        };
        if(!cars[1].position_updated) {
            cout<<"car 2 position is not detected!" << endl;
            cars[1].empty = true;
        };

        return positions;
    };
};

int main () {
    // Read video
    VideoCapture video("..\\..\\video\\2_cars_with_markers_drive_bump_demo.avi");
    // Read first frame
    Mat frame;
    if(!video.read(frame)) {
        cout << "could not read frame" << endl;
        return -1;
    };

    vector<CarPosition> positions;

    Tracker tracker(frame);

    while (true) {
        // Read a new frame
        bool ok = video.read(frame);
        if (!ok) {
            break;
        };

        auto start = std::chrono::high_resolution_clock::now();

        positions = tracker.car_position_detecting(&frame);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "time: " << elapsed_seconds.count() << endl;
    };
    return 0;
}
