/**
 * @file gyroscope.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief 陀螺仪数据
 * @version 1.0
 * @date 2023-01-12
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <opencv2/core/types.hpp>

#include <string>
#include <vector>

//! @addtogroup dataio
//! @{

//! 陀螺仪数据
struct GyroData
{
    //! 移动姿态信息
    struct Translation
    {
        float x = 0.f;  //!< x 方向位置、距离（向右运动为正）\f$p_x\f$
        float y = 0.f;  //!< y 方向位置、距离（向下运动为正）\f$p_y\f$
        float z = 0.f;  //!< z 方向位置、距离（向前运动为正）\f$p_z\f$
        float vx = 0.f; //!< x 方向速度（向右运动为正）\f$v_x\f$
        float vy = 0.f; //!< y 方向速度（向下运动为正）\f$v_y\f$
        float vz = 0.f; //!< z 方向速度（向前运动为正）\f$v_z\f$
    } translation;

    //! 转动姿态信息
    struct Rotation
    {
        float yaw = 0.f;         //!< 偏转角（向右运动为正）
        float pitch = 0.f;       //!< 俯仰角（向下运动为正）
        float roll = 0.f;        //!< 滚转角（顺时针运动为正）
        float yaw_speed = 0.f;   //!< 偏转角速度（向右运动为正）
        float pitch_speed = 0.f; //!< 俯仰角速度（向下运动为正）
        float roll_speed = 0.f;  //!< 滚转角速度（顺时针运动为正）
    } rotation;

    /**
     * @brief 将陀螺仪数据写入 YAML 文件中
     * @note 默认以 `APPEND` 模式进行写入到结构体 `gyro_data_<?>` 中，`<?>` 表示写入的结构体标号，即参数 `idx`
     *
     * @param[in] path 写入的文件路径
     * @param[in] idx 写入的结构体标号
     * @param[in] data 待写入的陀螺仪数据
     * @return 是否写入成功
     */
    static bool write(const std::string &path, uint32_t idx, const GyroData &data) noexcept;

    /**
     * @brief 从指定 YAML 文件中读取陀螺仪数据
     * @note 访问指定下标的数据结构体 `gyro_data_<?>`，`<?>` 表示结构体标号，即参数 `idx`
     *
     * @param[in] path 读取的文件路径
     * @param[in] idx 结构体标号
     * @param[out] data 读取出的陀螺仪数据，读取失败则不对 `data` 做任何操作
     * @return 是否读取成功
     */
    static bool read(const std::string &path, uint32_t idx, GyroData &data) noexcept;
};

/**
 * @brief 统一通信传递数据
 * @deprecated 在 `version: 4.x` 中应考虑移除
 * @note 各类数据信息的顺序应与进程间、设备间通信协议中相同类型的数据信息一致
 */
struct TransferData : GyroData
{
    std::vector<uint8_t> data_u8; //!< 8 位整型数据信息
    std::vector<float> data_f32;  //!< 32 位浮点型数据信息

    /**
     * @brief 将通信传递数据写入 YAML 文件中
     * @note 默认以 APPEND 模式进行写入到结构体 `transfer_data_<?>` 中，`<?>` 表示写入的结构体标号，即参数 `idx`
     *
     * @param[in] path 写入的文件路径
     * @param[in] idx 写入的结构体标号
     * @param[in] data 待写入的通信传递数据
     * @return 是否写入成功
     */
    static bool write(const std::string &path, uint32_t idx, const TransferData &data) noexcept;

    /**
     * @brief 从指定 YAML 文件中读取通信传递数据
     * @note 访问指定下标的数据结构体 `transfer_data_<?>`，`<?>` 表示结构体标号，即参数 `idx`
     *
     * @param[in] path 读取的文件路径
     * @param[in] idx 结构体标号
     * @param[out] data 读取出的通信传递数据，读取失败则不对 `data` 做任何操作
     * @return 是否读取成功
     */
    static bool read(const std::string &path, uint32_t idx, TransferData &data) noexcept;
};

/// @example samples/tutorial_code/dataio/sample_read_corners.cpp 角点数据读取例程
/// @example samples/tutorial_code/dataio/sample_write_corners.cpp 角点数据写入例程

/**
 * @brief 将角点数据写入 YAML 文件中
 * @note
 * - 默认以 `APPEND` 模式进行写入到结构体 `corners_<?>` 中，`<?>` 表示写入的标号，即参数 `idx`
 * - YAML 文件参见 @ref readCorners
 *
 * @param[in] path 写入的文件路径
 * @param[in] idx 写入的标号
 * @param[in] corners 待写入的角点数据
 * @return 是否写入成功
 */
bool writeCorners(const std::string &path, uint32_t idx, const std::vector<std::vector<cv::Point2d>> &corners);

/**
 * @brief 从指定 YAML 文件中读取角点数据
 * @note
 * - 访问指定下标的数据结构体 `corners_<?>`，`<?>` 表示对应的标号，即参数 `idx`
 * - YAML 文件形如以下内容
 * @code {.yml}
 * %YAML:1.0
 * ---
 * frame_1:
 *    -
 *       -
 *          x: 1.9
 *          y: 2.11
 *    -
 *       -
 *          x: 3.12
 *          y: 4.13
 *       -
 *          x: 5.14
 *          y: 6.15
 * ...
 * ---
 * frame_2:
 *    -
 *       -
 *          x: 7.16
 *          y: 8.17
 * @endcode
 *
 * @param[in] path 读取的文件路径
 * @param[in] idx 结构体标号
 * @param[out] corners 读取出的角点数据，读取失败则不对 `data` 做任何操作
 * @return 是否写入成功
 */
bool readCorners(const std::string &path, uint32_t idx, std::vector<std::vector<cv::Point2d>> &corners);

//! @} dataio
