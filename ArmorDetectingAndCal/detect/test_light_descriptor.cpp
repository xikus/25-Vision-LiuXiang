#include <iostream>
#include <opencv2/opencv.hpp>
#include "LightDescriptor.h"
#include "detect_light.h"

using namespace cv;
using namespace std;


int main()
{
    Mat brightImg, grayImg, showImg;
    vector<vector<Point> > lightContours;
    vector<LightDescriptor> lightInfos;
    
    //读取视频并采样
    string videoPath = "/home/yoda/Downloads/zimiao_test.mp4";
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

        //绘制图片
        showImg = frame.clone();

        paintContours(showImg, lightInfos);

        //显示处理后的视频
        imshow("contours", showImg);
        waitKey(50);

        //lightInfos清零
        lightInfos.clear();
    }

    return 0;
}