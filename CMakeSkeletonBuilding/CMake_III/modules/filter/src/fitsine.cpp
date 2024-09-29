/**
 * @file fitsine.cpp
 * @author 黄裕炯 (961352855@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-09
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <memory>
#include <Eigen/LU>

#include "srvl/filter/fitsine.h"

//! FLOAT_MAX in different platform
#ifdef MAXFLOAT
constexpr static double FLOAT_MAX = static_cast<double>(MAXFLOAT);
#elif defined HUGE
constexpr static double FLOAT_MAX = static_cast<double>(HUGE);
#endif

constexpr static double PI = 3.14159265358979323;

using namespace Eigen;
using namespace std;

/**
 * @details 后续优化的思路: 1. 如果输入数组大小固定则将 eigen 矩阵声明为静态矩阵，利用define
 *                             或者 constexpr 来保证便利性
 *                          2. 源码编译 Eigen3 打开 intel CPU 专属加速编译功能
 *                          3. intel MKL 或者其他的矩阵运算库
 *                          4. 已经没有优化的必要了，除非出现更难的任务
 * @brief 目前的代码性能，单核运行 300 帧数据拟合 <0.1ms，完全足够满足击打神符的要求，不再花时间优化
 * @note 如果打开多线程加速功能，会影响其他代码的运行质量，所以我只考虑单核使用
 * */

/**
 * @brief 正弦拟合 频域滤波分辨率设置+中通滤波
 *
 * @param CenterFreq 频率通带中值
 * @param GridSize 区间数量
 * @param FreResolution 区间分辨率
 */
static MatrixXf GenerateFrequencyGrid(const float &CenterFreq, const int GridSize, const float &FreResolution)
{
    MatrixXf result(GridSize, 1);
    //奇数检验 保证CenterFreq在中间
    unsigned int oddGridSize = GridSize % 2 == 0 ? GridSize : GridSize + 1;
    for (int i = 0; i < GridSize; i++)
    {
        float freq = CenterFreq + (i - (int)((oddGridSize - 1) / 2)) * FreResolution;
        result(i, 0) = freq;
    }
    return result;
}

array<float, 4> fourParaFitEigen(size_t datacount,
                                 float *data_arr,
                                 const float &Fs,
                                 const float &CWFreq,
                                 const float &FreResolution,
                                 const uint32_t &GridSize)
{
    //数据一维矩阵 Data
    Matrix<float, Dynamic, 1> Data(datacount, 1);
    memcpy(Data.data(), data_arr, datacount * sizeof(float));
    //数据一维矩阵转置
    //    MatrixXf DataT = Data.transpose();
    //数据三维矩阵 DataPro
    Matrix<float, Dynamic, 3> Pro(datacount, 3);
    memcpy(Pro.data(), data_arr, datacount * sizeof(float));
    MatrixXf FrequencyGrid;
    FrequencyGrid = GenerateFrequencyGrid(CWFreq, GridSize, FreResolution);

    //频率-信号质量  二阶
    float maxFreq = 1.884;
    float maxValue = -FLOAT_MAX;
    float temp = -FLOAT_MAX;
    for (uint32_t freq_index = 0; freq_index < GridSize; ++freq_index)
    {
        float freq = FrequencyGrid(freq_index, 0);
        for (int row = 0; row < Pro.rows(); ++row)
        {
            float omega = 2 * PI * freq / (float)Fs;
            Pro(row, 0) = cos(omega * row);
            Pro(row, 1) = sin(omega * row);
            Pro(row, 2) = 1;
        }
        // // 计算注释
        // MatrixXf PT = Pro.transpose();
        // MatrixXf PTP = PT * Pro;
        // MatrixXf PTPInv = PTP.inverse();
        // MatrixXf PTPInvPT = PTPInv * PT;
        // MatrixXf PPTPInvPT = Pro * PTPInvPT;
        // MatrixXf PPTPInvPTD = PPTPInvPT * Data;
        // MatrixXf DTPPTPInvPTD = DataT * PPTPInvPTD;
        // g_omega(freq_index, 0) = DTPPTPInvPTD(0, 0);
        temp = (Data.transpose() * Pro * (Pro.transpose() * Pro).inverse() * Pro.transpose() * Data)(0, 0);
        if (temp > maxValue)
        {
            maxValue = temp;
            maxFreq = freq;
        }
    }

    //运算 三参数方法
    for (int r = 0; r < Pro.rows(); r++)
    {
        float omega = 2 * PI * maxFreq / ((float)Fs);
        Pro(r, 0) = cos(omega * r);
        Pro(r, 1) = sin(omega * r);
        Pro(r, 2) = 1;
    }

    // // 运算注释
    // MatrixXf PT = Pro.transpose();
    // MatrixXf PTP = PT * Pro;
    // MatrixXf PTPInv = PTP.inverse();
    // MatrixXf PTPInvPT = PTPInv * PT;
    MatrixXf max_gomegaFitted = (Pro.transpose() * Pro).inverse() * Pro.transpose() * Data;

    std::array<float, 4> result;
    result[0] = max_gomegaFitted(0, 0);
    result[1] = max_gomegaFitted(1, 0);
    result[2] = max_gomegaFitted(2, 0);
    result[3] = maxFreq;
    return result;
}
