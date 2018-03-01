#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

using namespace cv;
using namespace std;

struct Car{
    bool empty;
    Mat contour;
};

int main() {
    // Read video
    VideoCapture video("C:\\Users\\Nikolay Limonov\\Documents\\GitHub\\carproject\\video\\2_cars_with_markers_drive_"
                               "bump_demo.avi");

//    for(int l=0;l<100;l++){
//        Mat frame;
//        if (!video.read(frame)) {
//            cout << "could not read frame" << endl;
//            return -1;
//        };
//    };

    // Read first frame
    Mat frame;
    if (!video.read(frame)) {
        cout << "could not read frame" << endl;
        return -1;
    };

    Mat frame_track(frame.size(), CV_8UC1);
    cvtColor(frame, frame_track, CV_BGR2GRAY);

    Car car_1;
    car_1.empty = true;
    Car car_2;
    car_2.empty = true;
    Car cars[2];
    cars[0] = car_1;
    cars[1] = car_2;
    bool car_1_position_updated = false;
    bool car_2_position_updated = false;
    bool car_detected = false;

    Mat imgray(frame.size(), CV_8UC1);

    cvLoadImage("track_mask.png", CV_LOAD_IMAGE_ANYDEPTH);
    Mat track_mask(frame.size(), CV_8UC1);
    track_mask = imread("track_mask.png", 0);
    imwrite("track_mask_c++.png", track_mask);
    Mat imgray_2(frame.size(), CV_8UC1);
    Mat mask = (track_mask == 255);

    while(true){
        auto start = std::chrono::system_clock::now();

        // Read a new frame
        bool ok = video.read(frame);
        if (!ok){
            break;
        };

        cvtColor(frame, imgray, CV_BGR2GRAY);

        threshold(imgray, imgray, 60, 255, 0);

        imgray.copyTo(imgray_2, track_mask);
        imgray_2.copyTo(imgray);

        Mat_<int> kernel(5,5);
        for(int a = 0; a < kernel.rows; a++)
            for(int b = 0; b < kernel.cols; b++)
                kernel(a,b) = 1;

        // MORPH_OPEN = 2
        morphologyEx(imgray, imgray, 2, kernel);

        for(int a = 0; a < kernel.rows; a++)
            for(int b = 0; b < kernel.cols; b++)
                kernel(a,b) = 1;

        // MORPH_CLOSE = 3
        morphologyEx(imgray, imgray, 3, kernel);

        vector<Mat> contours;
        findContours(imgray, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // loop over the contours
        int i = 0;
        car_1_position_updated = false;
        car_2_position_updated = false;
        car_detected = false;
        for(int j = 0; j < contours.size(); j += 1){
            RotatedRect rect = minAreaRect(contours[j]);

            int length = max(int(rect.size.height), int(rect.size.width));
            int width = min(int(rect.size.height), int(rect.size.width));

            int min_length = 50;
            int max_length = 120;
            int min_width = 40;
            int max_width = 70;

            if (length < min_length || length > max_length || width < min_width || width > max_width) {
                continue;
            };

            car_detected = true;

            double cX = rect.center.x;
            double cY = rect.center.y;

            for(int k=0; k<2; k++){
                if(cars[k].empty == false){
                    RotatedRect car_rect = minAreaRect(cars[k].contour);
                    double car_X = car_rect.center.x;
                    double car_Y = car_rect.center.y;
                    double distance = sqrt((car_X - cX) * (car_X - cX) + (car_Y - cY) * (car_Y - cY));
                    if(distance < 50) {
                        if(k==0) {
                            car_1_position_updated = true;
                        } else if (k==1){
                            car_2_position_updated = true;
                        };
                        cars[k].contour = contours[j];
                        cars[k].empty = false;
                    };
                } else if(cars[k].empty == true){
                    if(k == 0){
                        if(cars[1].empty == false){
                            RotatedRect check_rect = minAreaRect(cars[1].contour);
                            double check_X = check_rect.center.x;
                            double check_Y = check_rect.center.y;
                            if(int(check_X) == int(cX) && int(check_Y) == int(cY)){
                                continue;
                            };
                        };
                        car_1_position_updated = true;
                    } else if(k==1) {
                        if(cars[0].empty == false){
                            RotatedRect check_rect = minAreaRect(cars[0].contour);
                            double check_X = check_rect.center.x;
                            double check_Y = check_rect.center.y;
                            if(int(check_X) == int(cX) && int(check_Y) == int(cY)){
                                continue;
                            };
                        };
                        car_2_position_updated = true;
                    };
                    cars[k].contour = contours[j];
                    cars[k].empty = false;
                };
            };

            i++;
            if(i == 1) {
                for (int m = 0; m < 2; m++) {
                    if (cars[m].empty == false) {
                        RotatedRect car_rect = minAreaRect(cars[m].contour);
                        double car_X = car_rect.center.x;
                        double car_Y = car_rect.center.y;
                        Point2f rect_corners[4];
                        car_rect.points(rect_corners);
                        cout << "Coordinates of car " << (m + 1) << ":" << car_X << ", " << car_Y << ", "
                             << car_rect.angle << endl;
                        for (int p = 0; p < 4; p++) {
                            line(imgray, rect_corners[p], rect_corners[(p + 1) % 4], 50 + m * 100, 5, 8);
                        };
                        circle(imgray, rect.center, 3, (50, 50, 50), -1);
                        putText(imgray, "centre", rect.center, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2);
                    };
                };
            };
        };

        if(car_1_position_updated == false){
            cout<<"car 1 position is not detected!" << endl;
            cars[0].empty = true;
        };
        if(car_2_position_updated == false){
            cout<<"car 2 position is not detected!" << endl;
            cars[1].empty = true;
        };
        fflush(stdout);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout<<"time:"<<elapsed_seconds.count()<<endl;

        resize(imgray, imgray, Size(int(1280 * 0.7), int(1024 * 0.7)));
        imshow("Image", imgray);
        waitKey(1);
    };

    return 0;
}
