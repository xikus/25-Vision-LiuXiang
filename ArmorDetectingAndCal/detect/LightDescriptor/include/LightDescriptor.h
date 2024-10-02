#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

class LightDescriptor
{
public:
    LightDescriptor();
    LightDescriptor(const cv::RotatedRect& light);
    ~LightDescriptor();
    const LightDescriptor& operator = (const LightDescriptor& ld);
    cv::RotatedRect rec() const;
    float width;
    float length;
    cv::Point2f center;
    float angle;
    float area;
};
