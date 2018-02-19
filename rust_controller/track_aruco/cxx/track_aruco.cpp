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
    unsigned int numMarkers,
    const unsigned char* const frame,
    unsigned int width,
    unsigned int height,
    int debug_print,
    Homography2* out
) {
    Mat frameMat = Mat(height, width, CV_8U, (void*)frame);

    vector<vector<Point3f> > markerCorners;
    vector<int> markerIds;

    for (int i = 0; i < numMarkers; i++) {
        Marker m = markers[i];
        vector<Point3f> corners;
        corners.push_back(Point3f(m.x, m.y, 0));
        corners.push_back(Point3f(m.x, m.y + m.width, 0));
        corners.push_back(Point3f(m.x + m.width, m.y + m.width, 0));
        corners.push_back(Point3f(m.x + m.width, m.y, 0));
        markerCorners.push_back(corners);
        markerIds.push_back(m.id);
    }

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
    Ptr<aruco::Board> board = aruco::Board::create(markerCorners, dictionary, markerIds);

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    detectorParams->adaptiveThreshWinSizeMin = 3;
    detectorParams->adaptiveThreshWinSizeMax = 23;
    detectorParams->adaptiveThreshWinSizeStep = 2;
    detectorParams->minMarkerPerimeterRate = 0.005;
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;

    vector<int> ids;
    vector<vector<Point2f> > corners, rejectedCorners;

    // detect markers
    aruco::detectMarkers(frameMat, dictionary, corners, ids, detectorParams, rejectedCorners);

    if (debug_print) {
        cout << "detected " << ids.size() << " ArUco markers" << endl;
    }

    // refind strategy to detect more markers
    aruco::refineDetectedMarkers(frameMat, board, corners, ids, rejectedCorners);

    if (debug_print) {
        cout << "refined to " << ids.size() << " ArUco markers" << endl;
    }

    // Doesn't work on OS X as windows need to be created and run from the main thread.
    //if (debug_print) {
    //    Mat frameColour = Mat(height, width, CV_8UC3);
    //    cv::cvtColor(frameMat, frameColour, cv::COLOR_GRAY2BGR);
    //    aruco::drawDetectedMarkers(frameColour, corners, ids);
    //    imshow("Detected ArUco Markers", frameColour);
    //    waitKey(0);
    //}

    vector<Point3f> worldPoints3d;
    vector<Point2f> imagePoints;
    aruco::getBoardObjectAndImagePoints(board, corners, ids, worldPoints3d, imagePoints);

    if (debug_print) {
        cout << "detected " << imagePoints.size() << " corners" << endl;
    }

    if (imagePoints.size() <= 4) {
        return imagePoints.size();
    }

    vector<Point2f> worldPoints2d;
    for (int i = 0; i < worldPoints3d.size(); i++) {
        Point3f point = worldPoints3d[i];
        worldPoints2d.push_back(Point2f(point.x, point.y));
    }

    Mat H = findHomography(worldPoints2d, imagePoints, LMEDS);

    out->values[0] = H.at<double>(0, 0);
    out->values[1] = H.at<double>(0, 1);
    out->values[2] = H.at<double>(0, 2);
    out->values[3] = H.at<double>(1, 0);
    out->values[4] = H.at<double>(1, 1);
    out->values[5] = H.at<double>(1, 2);
    out->values[6] = H.at<double>(2, 0);
    out->values[7] = H.at<double>(2, 1);
    out->values[8] = H.at<double>(2, 2);

    return imagePoints.size();
}
