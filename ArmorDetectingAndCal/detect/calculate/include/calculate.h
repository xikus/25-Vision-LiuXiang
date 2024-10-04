#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float getDist(Mat& translation_matrix);
void drawNormalVector(Mat& frame, Mat& rotation_matrix, Point2f& anchorPoint);
void drawDist(Mat& frame, float dist, Point2f& anchorPoint);