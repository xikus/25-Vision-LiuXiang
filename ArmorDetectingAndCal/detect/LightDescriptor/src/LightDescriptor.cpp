#include "LightDescriptor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;



//定义灯条的特征
LightDescriptor::LightDescriptor() {}

LightDescriptor::LightDescriptor(const cv::RotatedRect& light)
{
    width = light.size.width;
    length = light.size.height;
    center = light.center;
    angle = light.angle;
    area = light.size.area();
}

LightDescriptor::~LightDescriptor() {}
const LightDescriptor& LightDescriptor::operator = (const LightDescriptor& ld)
{
    this->width = ld.width;
    this->length = ld.length;
    this->center = ld.center;
    this->angle = ld.angle;
    this->area = ld.area;
    return *this;
}

//返回旋转矩阵
cv::RotatedRect LightDescriptor::rec() const
{
    return cv::RotatedRect(center, cv::Size2f(width, length), angle);
}


