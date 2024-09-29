/**
 * @file paint.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief 绘画辅助工具
 * @version 1.0
 * @date 2023-01-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2023
 *
 */

#pragma once

#include <vector>

#include <opencv2/imgproc.hpp>

//! @addtogroup imgproc
//! @{

//! 阵列工具
class ArrayTool
{
public:
    /**
     * @brief 平面线性阵列
     *
     * @param sketch 阵列特征
     * @param spacing 阵列间距（Δx, Δy）
     * @param times 阵列个数（包括原特征）
     * @return 阵列结果
     */
    template <typename _Tp>
    static std::vector<std::vector<cv::Point_<_Tp>>> linear2D(const std::vector<cv::Point_<_Tp>> &sketch,
                                                              const cv::Point_<_Tp> &spacing, std::size_t times)
    {
        // 数据准备
        std::vector<std::vector<cv::Point_<_Tp>>> sketchs;
        sketchs.reserve(times);
        sketchs.push_back(sketch);
        // 循环阵列
        for (std::size_t i = 1; i < times; ++i)
        {
            std::vector<cv::Point_<_Tp>> tmp;
            tmp.reserve(sketch.size());
            for (auto &point : sketch)
                tmp.emplace_back(point.x + spacing.x * i, point.y + spacing.y * i);
            sketchs.emplace_back(tmp);
        }
        return sketchs;
    }

    /**
     * @brief 平面圆周阵列
     *
     * @param sketch 阵列特征
     * @param center 阵列中心
     * @param spacing 阵列间距（Δθ，角度制，像素坐标系）
     * @param times 阵列个数（包括原特征）
     * @return 阵列结果
     */
    template <typename _Tp>
    static std::vector<std::vector<cv::Point_<_Tp>>> circular2D(const std::vector<cv::Point_<_Tp>> &sketch,
                                                                const cv::Point_<_Tp> &center, double spacing, std::size_t times)
    {
        // 数据准备
        std::vector<std::vector<cv::Point_<_Tp>>> sketchs;
        sketchs.reserve(times);
        sketchs.push_back(sketch);
        // 循环阵列
        for (size_t i = 1; i < times; ++i)
        {
            // 计算旋转矩阵，左手系
            double spacing_rad = spacing * static_cast<double>(i) * M_PI / 180.0;
            double c_r = cos(spacing_rad);
            double s_r = sin(spacing_rad);
            cv::Matx22d R = {c_r, -s_r,
                             s_r, c_r};
            std::vector<cv::Point_<_Tp>> tmp;
            tmp.reserve(sketch.size());
            for (auto &point : sketch)
            {
                cv::Vec2d vec = {static_cast<double>(point.x - center.x),
                                 static_cast<double>(point.y - center.y)};
                vec = R * vec;
                tmp.emplace_back(static_cast<_Tp>(center.x + vec(0)),
                                 static_cast<_Tp>(center.y + vec(1)));
            }
            sketchs.emplace_back(tmp);
        }
        return sketchs;
    }

    /**
     * @brief 空间线性阵列
     *
     * @param sketch 阵列特征
     * @param spacing 阵列间距（Δx, Δy, Δz）
     * @param times 阵列个数（包括原特征）
     * @return 阵列结果
     */
    template <typename _Tp>
    static std::vector<std::vector<cv::Point3_<_Tp>>> linear3D(const std::vector<cv::Point3_<_Tp>> &sketch,
                                                               const cv::Point3_<_Tp> &spacing, std::size_t times)
    {
        std::vector<std::vector<cv::Point3_<_Tp>>> sketchs;
        sketchs.reserve(times);
        sketchs.push_back(sketch);
        for (std::size_t i = 1; i < times; ++i)
        {
            std::vector<cv::Point3_<_Tp>> tmp;
            tmp.reserve(sketch.size());
            for (auto &point : sketch)
                tmp.emplace_back(point.x + spacing.x * i,
                                 point.y + spacing.y * i,
                                 point.z + spacing.z * i);
            sketchs.emplace_back(tmp);
        }
        return sketchs;
    }

    /**
     * @brief 空间圆周阵列
     *
     * @param sketch 阵列特征
     * @param center 阵列中心
     * @param axis 阵列转轴
     * @param spacing 阵列间距（Δθ，角度制）
     * @param times 阵列个数（包括原特征）
     * @return 阵列结果
     */
    template <typename _Tp>
    static std::vector<std::vector<cv::Point3_<_Tp>>> circular3D(const std::vector<cv::Point3_<_Tp>> &sketch,
                                                                 const cv::Point3_<_Tp> &center, const cv::Vec3d &axis,
                                                                 double spacing, std::size_t times)
    {
        // 数据准备
        std::vector<std::vector<cv::Point3_<_Tp>>> sketchs;
        sketchs.reserve(times);
        sketchs.push_back(sketch);
        // 循环阵列
        for (size_t i = 1; i < times; ++i)
        {
            // 计算旋转矩阵
            double spacing_rad = spacing * static_cast<double>(i) * M_PI / 180.0;
            double c_r = cos(spacing_rad);
            double s_r = sin(spacing_rad);
            auto &a = axis;
            cv::Matx33d R = {c_r + (1 - c_r) * a(0) * a(0), (1 - c_r) * a(0) * a(1) - s_r * a(2), (1 - c_r) * a(0) * a(2) + s_r * a(1),
                             (1 - c_r) * a(0) * a(1) + s_r * a(2), c_r + (1 - c_r) * a(1) * a(1), (1 - c_r) * a(1) * a(2) - s_r * a(0),
                             (1 - c_r) * a(0) * a(2) - s_r * a(1), (1 - c_r) * a(1) * a(2) + s_r * a(0), c_r + (1 - c_r) * a(2) * a(2)};
            std::vector<cv::Point3_<_Tp>> tmp;
            tmp.reserve(sketch.size());
            for (auto &point : sketch)
            {
                cv::Vec3d vec = {static_cast<double>(point.x - center.x),
                                 static_cast<double>(point.y - center.y),
                                 static_cast<double>(point.z - center.z)};
                vec = R * vec;
                tmp.emplace_back(static_cast<_Tp>(center.x + vec(0)),
                                 static_cast<_Tp>(center.y + vec(1)),
                                 static_cast<_Tp>(center.z + vec(2)));
            }
            sketchs.emplace_back(tmp);
        }
        return sketchs;
    }
};

//! @}
