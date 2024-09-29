/**
 * @file filter.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 
 * @version 1.0
 * @date 2022-11-03
 * 
 * @copyright Copyright SCUT RobotLab(c) 2022
 * 
 */

#pragma once

/**
 * @defgroup filter 滤波模块
 * @{
 *     @defgroup kalman_filter 卡尔曼滤波 
 *     @defgroup fitsine_filter 四参数正弦拟合
 * @}
 */

#include <srvl/srvl_modules.hpp>

#ifdef HAVE_SRVL_FILTER
#include "filter/fitsine.h"
#include "filter/kalman.hpp"
#endif //! HAVE_SRVL_FILTER
