#include <iostream>
#include <opencv2/opencv.hpp>
#include "LightDescriptor.h"
#include "match_light.h"


using namespace std;
using namespace cv;

float light_max_angle_diff_ = 6; //灯条最大角度差10
float light_max_height_diff_ratio_ = 0.8; //灯条最大高度差比率
float light_max_y_diff_ratio_ = 1.2; //灯条最大y差比率
float light_min_x_diff_ratio_ = 0.6; //灯条最小x差比率
float light_max_x_diff_ratio_ = 2; //灯条最大x差比率
float armor_max_aspect_ratio_ = 3; //装甲板最大长宽比5
float armor_min_aspect_ratio_ = 0.04; //装甲板最小长宽比0.1
float lendiff_max = 0.5;

//
void matchLight(vector<LightDescriptor>& lightInfos, vector<vector<int>>& armorInfos) {
    //将灯条按照x坐标排序
    sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2) {
        return ld1.center.x < ld2.center.x;
        });

    //遍历灯条
    for (size_t i = 0; i < lightInfos.size(); i++)
    {
        for (size_t j = i; j < lightInfos.size(); j++)
        {
            const LightDescriptor& rightLight = lightInfos[i];
            const LightDescriptor& leftLight = lightInfos[j];

            //计算灯条的角度差
            float angleDiff_ = abs(leftLight.angle - rightLight.angle);

            //长度差比率
            float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);

            //筛选
            if (angleDiff_ > light_max_angle_diff_ ||
                LenDiff_ratio > light_max_height_diff_ratio_)
            {
                continue;
            }

            //左右灯条相距距离
            float dis = sqrt(((rightLight.center.x - leftLight.center.x), 2) + powf((rightLight.center.y - leftLight.center.y), 2));

            //左右灯条长度的平均值
            float meanLen = (leftLight.length + rightLight.length) / 2;

            //左右灯条长度差比值
            float lendiff = abs(leftLight.length - rightLight.length) / meanLen;

            //左右灯条中心点y的差值
            float yDiff = abs(leftLight.center.y - rightLight.center.y);
            //左右灯条中心点x的差值
            float xDiff = abs(leftLight.center.x - rightLight.center.x);

            //y差比率.emplace_back
            float yDiff_ratio = yDiff / meanLen;
            //x差比率
            float xDiff_ratio = xDiff / meanLen;

            //相距距离与灯条长度比值
            float ratio = dis / meanLen;

            //筛选
            if (lendiff > lendiff_max ||
                yDiff_ratio > light_max_y_diff_ratio_ ||
                xDiff_ratio < light_min_x_diff_ratio_ ||
                xDiff_ratio > light_max_x_diff_ratio_ ||
                ratio > armor_max_aspect_ratio_ ||
                ratio < armor_min_aspect_ratio_)
            {
                continue;
            }
            vector<int> temp = { (int)i, (int)j };
            armorInfos.push_back(temp);
        }
    }
}

void drawArmor(const vector<vector<int>>& armorInfos, const vector<LightDescriptor>& lightInfos, Mat& src) {
    for (const auto& armor : armorInfos) {

        const LightDescriptor& leftRect = lightInfos[armor[0]];
        const LightDescriptor& rightRect = lightInfos[armor[1]];

        line(src, Point(leftRect.center.x, leftRect.center.y + leftRect.length / 2), Point(rightRect.center.x, rightRect.center.y + rightRect.length / 2), Scalar(0, 0, 255), 2);
        line(src, Point(rightRect.center.x, rightRect.center.y + rightRect.length / 2), Point(rightRect.center.x, rightRect.center.y - rightRect.length / 2), Scalar(0, 0, 255), 2);
        line(src, Point(rightRect.center.x, rightRect.center.y - rightRect.length / 2), Point(leftRect.center.x, leftRect.center.y - leftRect.length / 2), Scalar(0, 0, 255), 2);
        line(src, Point(leftRect.center.x, leftRect.center.y - leftRect.length / 2), Point(leftRect.center.x, leftRect.center.y + leftRect.length / 2), Scalar(0, 0, 255), 2);
        //line(src, leftRect.center, rightRect.center, Scalar(0, 255, 0), 2);
    }
}

vector<vector<Point2f>> getArmorVertex(const vector<vector<int>>& armorInfos, const vector<LightDescriptor>& lightInfos) {
    vector<vector<Point2f>> Armors;
    for (const auto& armor : armorInfos) {
        const LightDescriptor& leftRect = lightInfos[armor[0]];
        const LightDescriptor& rightRect = lightInfos[armor[1]];
        Point2f upLeft(leftRect.center.x, leftRect.center.y + leftRect.length / 2);
        Point2f upRight(rightRect.center.x, rightRect.center.y + rightRect.length / 2);
        Point2f downRight(rightRect.center.x, rightRect.center.y - rightRect.length / 2);
        Point2f downLeft(leftRect.center.x, leftRect.center.y - leftRect.length / 2);
        vector<Point2f> armorVertices;
        armorVertices.emplace_back(upLeft);
        armorVertices.emplace_back(upRight);
        armorVertices.emplace_back(downRight);
        armorVertices.emplace_back(downLeft);
        Armors.emplace_back(armorVertices);
    }
    return Armors; 
}