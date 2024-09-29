/**
 * @file protocol.h
 * @author 赵曦 (535394140@qq.com)
 * @brief 协议数据包管理头文件
 * @version 1.0
 * @date 2021-09-18
 *
 * @copyright Copyright SCUT RobotLab(c) 2021
 *
 */

#pragma once

#include "dataio.h"

//! @addtogroup dataio
//! @{

/**
 * @brief 抽象协议类
 * @deprecated 在 `version: 4.x` 中应考虑移除
 */
class protocol
{
public:
    protocol() = default;
    virtual ~protocol() = default;

    /**
     * @brief 读取数据
     */
    virtual TransferData read() { return {}; };

    /**
     * @brief 写入数据
     */
    virtual void write(const TransferData &) {};

    // 是否打开
    virtual bool isOpened() { return false; };
};

//! @} dataio
