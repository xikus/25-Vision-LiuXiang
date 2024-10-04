#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <LightDescriptor.h>

using namespace std;
using namespace cv;

void matchLight(vector<LightDescriptor>& lightInfos, vector<vector<int>>& armorInfos);
void drawArmor(const vector<vector<int>>& armorInfos, const vector<LightDescriptor>& lightInfos, Mat& src);
vector<vector<Point2f>> getArmorVertex(const vector<vector<int>>& armorInfos, const vector<LightDescriptor>& lightInfos);