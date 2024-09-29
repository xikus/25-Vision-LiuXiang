/**
 * @file version.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief SRVL 版本控制
 * @version 1.0
 * 
 * @copyright Copyright 2023 (c), zhaoxi
 * 
 */

#pragma once

#include <string>

#define SRVL_VERSION_MAJOR 3
#define SRVL_VERSION_MINOR 6
#define SRVL_VERSION_PATCH 0
#define SRVL_VERSION_STATUS "-dev"

#define SRVLAUX_STR_EXP(__A) #__A
#define SRVLAUX_STR(__A) SRVLAUX_STR_EXP(__A)

#define SRVL_VERSION SRVLAUX_STR(SRVL_VERSION_MAJOR) "." SRVLAUX_STR(SRVL_VERSION_MINOR) "." SRVLAUX_STR(SRVL_VERSION_PATCH) SRVL_VERSION_STATUS

namespace srvl
{

/**
 * @brief 返回库版本字符串
 * @note "For example: 3.6.0-dev"
 * @see getVersionMajor, getVersionMinor, getVersionPatch
 */
inline std::string getVersionString() { return SRVL_VERSION; }

//! 返回主要库版本
inline int getVersionMajor() { return SRVL_VERSION_MAJOR; }

//! 返回次要库版本
inline int getVersionMinor() { return SRVL_VERSION_MINOR; }

//! 返回库版本的修订字段
inline int getVersionPatch() { return SRVL_VERSION_PATCH; }

} // namespace srvl
