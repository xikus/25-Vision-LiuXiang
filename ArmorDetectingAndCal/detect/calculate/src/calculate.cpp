#include <calculate.h>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


const Mat intrinsic_matrix = (Mat_<float>(3, 3) << 1.3859739625395162e+03, 0., 9.3622464596653492e+02, 0., 1.3815353250336800e+03, 4.9459467170828475e+02, 0., 0., 1.);

const Mat dist_coeffs = (Mat_<float>(1, 5) << 0.000000, 0.000000, 0.000000, 0.000000, 0.000000);

const std::vector<cv::Point3f> truepoints_little = {
                                            cv::Point3f(-0.0675,0.03125,0),
                                            cv::Point3f(-0.0675,-0.03125,0),
                                            cv::Point3f(0.0675,-0.03125,0),
                                            cv::Point3f(0.0675,0.03125,0)
};

const std::vector<cv::Point3f> truepoints_big = {
                                    cv::Point3f(-0.115,0.03125,0),
                                    cv::Point3f(-0.115,-0.03125,0),
                                    cv::Point3f(0.115,-0.03125,0),
                                    cv::Point3f(0.115,0.03125,0)
};

float getDist(Mat& translation_matrix) {
    return sqrt(pow(translation_matrix.at<double>(0, 0), 2) + pow(translation_matrix.at<double>(0, 1), 2) + pow(translation_matrix.at<double>(0, 2), 2));
}

void drawNormalVector(Mat& frame, Mat& rotation_matrix, Point2f& anchorPoint) {
    Mat rotation_matrix_inv = rotation_matrix.inv();
    Mat normalVector;
    rotation_matrix_inv.col(2).copyTo(normalVector);
    putText(frame, "Normal Vector: " + to_string(normalVector.at<double>(0, 0)) + " " + to_string(normalVector.at<double>(1, 0)) + " " + to_string(normalVector.at<double>(2, 0)), anchorPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);

}

// void drawDist(Mat& frame, float dist) {
//     putText(frame, "Distance: " + to_string(dist), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
// }

//将每一个检测框的距离标在框旁边
void drawDist(Mat& frame, float dist, Point2f& anchorPoint) {
    putText(frame, "Distance: " + to_string(dist), anchorPoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
}