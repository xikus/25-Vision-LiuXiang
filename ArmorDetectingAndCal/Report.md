![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

# 华南虎视觉组实习生任务装甲板识别

**实习生姓名：柳翔**

## 1. 代码思路

### 1.1 LightDescriptor
`这一部分提供灯条的数据结构`
- 改造自`RotatedRect`,提供`width`,`length`,`center`,`angle`,`area`,`rec()`等接口. 其中`rec()`返回对应RotatedRect.
---
### 1.2 detect_light
`这一部分是为了检测灯条`
- 预处理: 将每一个`frame`转化为3个单通道图片,运用通道相减法得到灯条的灰度图.接着通过阈值化,膨胀等操作预处理图像. 
- 筛选: 主要通过面积, 凸度, 宽高比筛选出灯条
- 展示: 在每一个`frame`上画出灯条轮廓
---
### 1.3 match_light
`这一部分是为了匹配灯条, 从而检测出装甲板`
- 排序: 将`detect_light`得到的`lightInfos`按x坐标排序
- 匹配:按照角度差, 长度差比率, 左右灯条相距距离, x差比率, y差比率, 相距距离与灯条长度比值等指标进行匹配
- 展示: 将得到的装甲板绘至`frame`
- 为`calculate`提供`getArmorVertex`接口
---
### 1.4 calculate 
`这一部分是为了计算装甲板到相机距离和装甲板法向量`
- 根据`match_light`提供的`getArmorVertex`接口得到装甲板的四角数据
- 距离:通过对`translation_matrix`进行计算从而得到装甲板到相机的距离
- 法向量: 通过对`rotation_matrix`进行计算得到装甲板法向量
- 补充: `translation_matrix`和`rotation_matrix`都可由`solvePnp`解出

## 2. 遇到问题

- 1. 除了通道相减法，想到可以转到hsv空间分离红色域(未尝试)
---
- 2. 通道相减能否使用G通道？(已解决)
    ```C++
    Mat grayImg;
    if (_enemy_color == RED) {
        grayImg = channels[2] - channels[0];//R-B,此时R很大,B接近0,得到的是红色物体的图像
    }
    else {
        grayImg = channels[0] - channels[2];//B-R,此时B很大,R接近0，得到的是蓝色物体的图像
    }
    return grayImg;
    ``` 
    ```可以的,但是效果不如B-R```
    
    ---
- 3. `findContours()`会改变输入图像,一般的解决方法是利用`clone()`复制出另外一张图输入`findContours()`里面
---
- 4. solvePnp()函数
```C++
bool solvePnP(InputArray objectPoints, InputArray imagePoints,
              InputArray cameraMatrix, InputArray distCoeffs,
              OutputArray rvec, OutputArray tvec,
              bool useExtrinsicGuess = false,
              int flags = SOLVEPNP_ITERATIVE);
//得到解算结果后，rvec为旋转矢量形式，后续计算不方便，所以一般会用Rodrigues公式转为旋转矩阵
Rodrigues(rvec, rotation_matrix);
```
> objectPoints：世界坐标系（上图中OwXwYwZw）下的3D点坐标数组
imagePoints：图像（上图中ouv）中对应3D点的成像点坐标数组
cameraMatrix：相机内参矩阵，3×3
distCoeffs：相机畸变系数数组，可以为NULL，此时视为无畸变。
rvec和tvec：计算结果输出，rvec为旋转向量，tvec为平移向量，两者合并表达的是物体整体（即世界坐标系）在相机坐标系中的位姿
## 3. 效果图
- 见演示视频
## 4. 总结
- 装甲板任务主要任务量来源于调节各种参数. 下一步改进为对参数进行进一步decouple和使用GUI工具使调节更加便利.

## 5. CMake框架搭建
```├── CHANGELOG.md
├── detect
│   ├── calculate
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── calculate.h
│   │   └── src
│   │       └── calculate.cpp
│   ├── CMakeLists.txt
│   ├── detect_light
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── detect_light.h
│   │   └── src
│   │       └── detect_light.cpp
│   ├── LightDescriptor
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── LightDescriptor.h
│   │   └── src
│   │       └── LightDescriptor.cpp
│   ├── match_light
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── match_light.h
│   │   └── src
│   │       └── match_light.cpp
│   ├── simplescreenrecorder-2024-10-06_16.27.09.mkv
│   ├── test_light_descriptor.cpp
│   └── zimiao_test.mp4
├── README.md
└── Report.md
```
