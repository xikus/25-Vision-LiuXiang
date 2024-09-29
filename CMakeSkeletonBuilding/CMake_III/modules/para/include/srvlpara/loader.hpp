/**
 * @file loader.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief 
 * @version 1.0
 * @date 2022-11-30
 * 
 * @copyright Copyright SCUT RobotLab(c) 2022
 * 
 */

#pragma once

//! @defgroup para 参数及加载模块

#include "srvl/core/util.hpp"

namespace para
{

//! @addtogroup para
//! @{

/**
 * @brief 参数加载
 * 
 * @tparam _Tp 参数类型
 * @param para_obj 参数对象
 * @param file_path 参数 yml 文件
 */
template <typename _Tp, typename Enable = typename _Tp::paraId>
inline void load(_Tp &para_obj, const std::string &file_path) { para_obj = _Tp(file_path); }

/**
 * @brief 参数读取，忽略为空的节点
 * 
 * @param n cv::FileNode 节点
 * @param t 目标数据
 */
template<typename _FileNode, typename _Tp>
inline void readExcludeNone(const _FileNode &n, _Tp &t) { n.isNone() ? void(0) : n >> t; } 

//! @} para

} // namespace para
