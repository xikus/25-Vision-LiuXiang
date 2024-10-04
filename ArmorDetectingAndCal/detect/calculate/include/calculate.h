#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

float getDist(Mat& translation_matrix);
void drawDist(Mat& frame, float dist);
void drawDist(Mat& frame, float dist, Point2f& anchorPoint);