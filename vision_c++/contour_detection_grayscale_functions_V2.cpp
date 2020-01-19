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
    bool position_updated = false;
    Mat contour;
    CarPosition position;
};

vector<CarPosition> position_update (Car* cars, int number_of_cars) {
    vector<CarPosition> positions;
    for (int k = 0; k < number_of_cars; k++) {
        if (cars[k].position_updated) {
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

void car_contour_filtering (Car* cars, int number_of_cars, Mat contour) {
    RotatedRect rect = minAreaRect(contour);

    auto length = int(fmax(rect.size.height, rect.size.width));
    auto width = int(fmin(rect.size.height, rect.size.width));

    // make them global consts
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
        if (cars[k].position_updated) {
            RotatedRect car_rect = minAreaRect(cars[k].contour);
            double dX = car_rect.center.x - cX;
            double dY = car_rect.center.y - cY;
            double distance = sqrt(dX*dX +  dY*dY);
            if (distance < 50) {
                cars[k].position_updated = true;
                cars[k].contour = contour;
            };
        } else {
            // Other car index. Works for 2 cars only.
            int m = (k + 1) % 2;
            if (cars[m].position_updated) {
                RotatedRect check_rect = minAreaRect(cars[m].contour);
                double dX = check_rect.center.x - cX;
                double dY = check_rect.center.y - cY;
                double distance = sqrt(dX*dX +  dY*dY);
                if (distance < 50) {
                    continue;
                };
            };
            cars[k].position_updated = true;
            cars[k].contour = contour;
        };
    };
};

Mat frame_filtering (Mat image, Mat track_mask, Mat_<int> kernel) {
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
    Mat track_mask;
    Mat kernel;
    vector<Mat> contours;

    Tracker (Mat mask) {
        Mat imgray(mask.size(), CV_8UC1);

        track_mask = mask;

        kernel = Mat::ones(5,5, CV_32S);
    };

    vector<CarPosition> car_position_detecting (Mat frame) {
        vector<CarPosition> positions;
        cvtColor(frame, imgray, CV_BGR2GRAY);

        imgray = frame_filtering(imgray, track_mask, kernel);

        findContours(imgray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        cars[0].position_updated = false;
        cars[1].position_updated = false;

        for(int j = 0; j < contours.size(); j += 1) {
            car_contour_filtering(cars, 2, contours[j]);
            if (j == (contours.size() - 1)) {
                positions = position_update(cars, 2);
            };
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
    Mat track_mask;
    track_mask = imread("..\\track_mask.png", 0);
    Tracker tracker(track_mask);

    while (video.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();

        positions = tracker.car_position_detecting(frame);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "time: " << elapsed_seconds.count() << endl;

        for (int i=0;i<positions.size();i++) {
            cout<<"Position of car "<<(i+1)<<": "<<positions[i].x<<", "<<positions[i].y<<", "
                <<positions[i].heading<<endl;
        };
    };
    return 0;
}
