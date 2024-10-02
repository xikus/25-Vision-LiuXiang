#pragma once
#include <opencv2/opencv.hpp>
#include "LightDescriptor.h"
#include <iostream>

using namespace cv;
using namespace std;

Mat separateColors(Mat& _roiImg);
void getBrightImg(Mat& _grayImg, Mat& _binBrightImg);
void adjustRec(RotatedRect& rec);
void filterContours(vector<vector<Point> >& lightContours, vector<LightDescriptor>& lightInfos);
void paintContours(Mat& srcImg, const vector<LightDescriptor>& lightInfos);

enum EnemyColor {
    RED,
    BLUE
};