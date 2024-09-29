/**
 * @file ml.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief 
 * @version 1.0
 * @date 2023-05-20
 * 
 * @copyright Copyright 2023 (c), zhaoxi
 * 
 */

#pragma once

/**
 * @defgroup ml 机器学习与深度学习支持库
 * @{
 *     @defgroup ml_ort ONNX-Runtime 分类网络部署库
 * @}
 */

#include <srvl/srvl_modules.hpp>

#ifdef HAVE_SRVL_ORT
#include "ml/ort.h"
#endif //! HAVE_SRVL_ORT
