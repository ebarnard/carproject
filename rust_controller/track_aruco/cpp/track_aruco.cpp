#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

typedef struct {
    unsigned int id;
    double x;
    double y;
    double width;
} Marker;

typedef struct {
    double values[9];
} Homography2;

extern "C" unsigned int find_homography(
    const Marker* const markers,
    unsigned int num_markers,
    const unsigned char* const frame,
    unsigned int width,
    unsigned int height,
    int debug_print,
    Homography2* H_out
) {
    Mat frame_mat = Mat(height, width, CV_8U, (void*)frame);

    vector<vector<Point3f> > marker_corners;
    vector<int> marker_ids;

    for (int i = 0; i < num_markers; i++) {
        Marker m = markers[i];
        vector<Point3f> corners;
        corners.push_back(Point3f(m.x, m.y, 0));
        corners.push_back(Point3f(m.x, m.y + m.width, 0));
        corners.push_back(Point3f(m.x + m.width, m.y + m.width, 0));
        corners.push_back(Point3f(m.x + m.width, m.y, 0));
        marker_corners.push_back(corners);
        marker_ids.push_back(m.id);
    }

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
    Ptr<aruco::Board> board = aruco::Board::create(marker_corners, dictionary, marker_ids);

    Ptr<aruco::DetectorParameters> detector_params = aruco::DetectorParameters::create();
    detector_params->adaptiveThreshWinSizeMin = 3;
    detector_params->adaptiveThreshWinSizeMax = 23;
    detector_params->adaptiveThreshWinSizeStep = 2;
    detector_params->minMarkerPerimeterRate = 0.005;
    detector_params->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;

    vector<int> ids;
    vector<vector<Point2f> > corners, rejected_corners;

    // detect markers
    aruco::detectMarkers(frame_mat, dictionary, corners, ids, detector_params, rejected_corners);

    if (debug_print) {
        cout << "detected " << ids.size() << " ArUco markers" << endl;
    }

    // refind strategy to detect more markers
    aruco::refineDetectedMarkers(frame_mat, board, corners, ids, rejected_corners);

    if (debug_print) {
        cout << "refined to " << ids.size() << " ArUco markers" << endl;
    }

    // Doesn't work on OS X as windows need to be created and run from the main thread.
    //if (debug_print) {
    //    Mat frame_colour = Mat(height, width, CV_8UC3);
    //    cv::cvtColor(frame_mat, frame_colour, cv::COLOR_GRAY2BGR);
    //    aruco::drawDetectedMarkers(frame_colour, corners, ids);
    //    imshow("Detected ArUco Markers", frame_colour);
    //    waitKey(0);
    //}

    vector<Point3f> world_points_3d;
    vector<Point2f> image_points;
    aruco::getBoardObjectAndImagePoints(board, corners, ids, world_points_3d, image_points);

    if (debug_print) {
        cout << "detected " << image_points.size() << " corners" << endl;
    }

    if (image_points.size() == 0) {
        return image_points.size();
    }

    vector<Point2f> world_points_2d;
    for (int i = 0; i < world_points_3d.size(); i++) {
        Point3f point = world_points_3d[i];
        world_points_2d.push_back(Point2f(point.x, point.y));
    }

    Mat H = findHomography(world_points_2d, image_points, LMEDS);

    H_out->values[0] = H.at<double>(0, 0);
    H_out->values[1] = H.at<double>(0, 1);
    H_out->values[2] = H.at<double>(0, 2);
    H_out->values[3] = H.at<double>(1, 0);
    H_out->values[4] = H.at<double>(1, 1);
    H_out->values[5] = H.at<double>(1, 2);
    H_out->values[6] = H.at<double>(2, 0);
    H_out->values[7] = H.at<double>(2, 1);
    H_out->values[8] = H.at<double>(2, 2);

    return image_points.size();
}
