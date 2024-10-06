#include <iostream>
#include <opencv2/opencv.hpp>
#include "LightDescriptor.h"
#include "detect_light.h"
#include "match_light.h"
#include "calculate.h"
#include<opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

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

const Mat intrinsic_matrix = (Mat_<float>(3, 3) << 1.3859739625395162e+03, 0., 9.3622464596653492e+02, 0., 1.3815353250336800e+03, 4.9459467170828475e+02, 0., 0., 1.);

const Mat dist_coeffs = (Mat_<float>(1, 5) << 0.000000, 0.000000, 0.000000, 0.000000, 0.000000);


int main()
{
    Mat brightImg, grayImg, showImg;
    vector<vector<Point> > lightContours;
    vector<LightDescriptor> lightInfos;
    vector<vector<int>> armorInfos;
    Mat translation_matrix, rotation_matrix, R;

    //读取视频并采样
    string videoPath = "../zimiao_test.mp4";
    //VideoWriter wrdist_coeffster("/home/yoda/25-Vision-LiuXiang/ArmorDetectingAndCal/detect/test1.mp4", 0x7634706d, 20, Size(1080, 754), true);
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        cout << "视频打开失败" << endl;
        return -1;
    }

    Mat frame;
    while (1)
    {
        cap >> frame;

        //分离颜色
        grayImg = separateColors(frame);

        //获取二值图
        getBrightImg(grayImg, brightImg);

        //寻找轮廓(findContours()会改变原图,所以传入clone())
        findContours(brightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //筛选灯条
        filterContours(lightContours, lightInfos);
        matchLight(lightInfos, armorInfos);

        //绘制图片
        showImg = frame.clone();
        
        paintContours(showImg, lightInfos);
        drawArmor(armorInfos, lightInfos, showImg);

        vector<vector<Point2f>> Armors = getArmorVertex(armorInfos, lightInfos);
        for (int i = 0; i < Armors.size(); i++)
        {
            solvePnP(truepoints_little, Armors[i], intrinsic_matrix, dist_coeffs, R, translation_matrix);
            Rodrigues(R, rotation_matrix);
            float dist = getDist(translation_matrix);
            drawDist(showImg, dist, Armors[i][0]);
            drawNormalVector(showImg, rotation_matrix, Armors[i][0]);
        }

        //显示处理后的视频
        imshow("armor", showImg);
        waitKey(50);
        //writer.write(showImg);

        //lightInfos清零
        lightInfos.clear();
        armorInfos.clear();
    }

    return 0;
}