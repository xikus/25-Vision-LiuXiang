#include "detect_light.h"
#include "LightDescriptor.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int light_min_area = 60;//灯条最小面积
double light_max_angle = 45.0;//灯条最大的倾斜角
double light_min_size = 5.0;//灯条最小尺寸
double light_contour_min_solidity = 0.5;//灯条最小凸度 注：凸度=轮廓面积/外接矩形面积
double light_max_ratio = 0.9;//灯条最大长宽比

EnemyColor _enemy_color = BLUE;

//分离出灯条，生成灰度图
Mat separateColors(Mat& _roiImg) {
    //把一个3通道图像转换成3个单通道图像
    vector<Mat> channels;
    split(_roiImg, channels);

    //剔除我们不想要的颜色
    //对于图像中红色的物体来说，其rgb分量中r的值最大，g和b在理想情况下应该是0，同理蓝色物体的b分量应该最大,将不想要的颜色减去，剩下的就是我们想要的颜色
    Mat grayImg;
    if (_enemy_color == RED) {
        grayImg = channels[2] - channels[0];//R-B,此时R很大,B接近0,得到的是红色物体的图像
    }
    else {
        grayImg = channels[0] - channels[2];//B-R,此时B很大,R接近0，得到的是蓝色物体的图像
    }
    return grayImg;
}

//输入灰度图,输出二值图
void getBrightImg(Mat& _grayImg, Mat& _binBrightImg) {
    //设置阈值，根据相机拍摄实际情况调整
    int brightness_low_threshold = 120;

    //阈值化
    threshold(_grayImg, _binBrightImg, brightness_low_threshold, 255, THRESH_BINARY);

    //设置膨胀卷积核
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    //膨胀
    dilate(_binBrightImg, _binBrightImg, element);
}

//调节灯条的角度，将其约束为-45~45°
void adjustRec(RotatedRect& rec) {
    float& angle = rec.angle;

    angle = angle > 45.0 ? angle - 90.0 : angle;
    angle = angle < -45.0 ? angle + 90.0 : angle;
}

//lightContours:存储被检测到的轮廓，lightInfos:存储经过筛选后的灯条info
void filterContours(vector<vector<Point> >& lightContours, vector<LightDescriptor>& lightInfos) {
    for (const auto& contour : lightContours) {
        //得到面积
        float lightContourArea = contourArea(contour);

        //面积太小的不要
        if (lightContourArea < light_min_area) continue;

        //椭圆拟合区域得到外接矩形注：凸度=轮廓面积/外接矩形面积
        RotatedRect lightRec = fitEllipse(contour);

        //矫正灯条的角度，将其约束为-45~45°(作用存疑)
        // adjustRec(lightRec);

        //宽高比、凸度筛选灯条
        if (lightRec.size.width / lightRec.size.height > light_max_ratio ||
            lightContourArea / lightRec.size.area() < light_contour_min_solidity)
            continue;

        //对灯条范围适当扩大(作用存疑)
        // lightRec.size.width *= light_color_detect_extend_ratio;
        // lightRec.size.height *= light_color_detect_extend_ratio;

        //直接将灯条保存
        lightInfos.push_back(LightDescriptor(lightRec));
    }
}

//画出灯条轮廓
void paintContours(Mat& srcImg, const vector<LightDescriptor>& lightInfos) {
    for (const auto& lightInfo : lightInfos) {
        RotatedRect lightRec = lightInfo.rec();
        Point2f vertex[4];
        lightRec.points(vertex);
        for (size_t i = 0; i < 4; i++) {
            line
            (srcImg,
                vertex[i],
                vertex[(i + 1) % 4],
                Scalar(0, 255, 0),
                1,
                LINE_AA);
        }
    }
}
