/**
 * @file util.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-11-03
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#ifndef __cplusplus
#error core.hpp header must be compiled as C++
#endif

#include <srvl/srvl_modules.hpp>

#include <cstdio>
#include <string>

//! @addtogroup core
//! @{

#define HIGHLIGHT_(msg...)                   \
    do                                       \
    {                                        \
        printf("\033[35m[   INFO   ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define WARNING_(msg...)                     \
    do                                       \
    {                                        \
        printf("\033[33m[   WARN   ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define PASS_(msg...)                        \
    do                                       \
    {                                        \
        printf("\033[32m[   PASS   ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define ERROR_(msg...)                       \
    do                                       \
    {                                        \
        printf("\033[31m[   ERR    ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define INFO_(msg...)                \
    do                               \
    {                                \
        printf("[   INFO   ] " msg); \
        printf("\n");                \
    } while (false)

#ifdef NDEBUG
#define DEBUG_WARNING_(msg...) ((void)0)
#define DEBUGOR_(msg...) ((void)0)
#define DEBUG_HIGHLIGHT_(msg...) ((void)0)
#define DEBUG_INFO_(msg...) ((void)0)
#define DEBUG_PASS_(msg...) ((void)0)
#else
#define DEBUG_WARNING_(msg...) WARNING_(msg)
#define DEBUGOR_(msg...) ERROR_(msg)
#define DEBUG_HIGHLIGHT_(msg...) HIGHLIGHT_(msg)
#define DEBUG_INFO_(msg...) INFO_(msg)
#define DEBUG_PASS_(msg...) PASS_(msg)
#endif

typedef int SRVLErrorCode; //!< 重定义 `int` 为 SRVLErrorCode

//! @brief SRVL 错误码
enum : SRVLErrorCode
{
    SRVL_StsOk = 0,           //!< 没有错误 No Error
    SRVL_StsBackTrace = -1,   //!< 回溯 Backtrace
    SRVL_StsError = -2,       //!< 未指定（未知）错误 Unspecified (Unknown) error
    SRVL_StsNoMem = -3,       //!< 内存不足 Insufficient memory
    SRVL_StsBadArg = -4,      //!< 参数异常 Bad argument
    SRVL_StsBadSize = -5,     //!< 数组大小不正确 Incorrect size of the array
    SRVL_StsNullPtr = -6,     //!< 空指针 Null pointer
    SRVL_StsDivByZero = -7,   //!< 发生了除以 `0` 的情况 Division by zero occurred
    SRVL_StsOutOfRange = -8,  //!< 其中一个参数的值超出了范围 One of the arguments' values is out of range
    SRVL_StsAssert = -9,      //!< 断言失败 Assertion failed
    SRVL_StsInvFmt = -10,     //!< 无效格式 Invalid format
    SRVL_BadDynamicType = -11 //!< 动态类型转换错误 Bad dynamic_cast type
};

const char *srvlErrorStr(SRVLErrorCode status);

namespace srvl
{

/**
 * @brief 返回使用类 `printf` 表达式格式化的文本字符串。
 * @note 该函数的作用类似于 `sprintf`，但形成并返回一个 STL 字符串。它可用于在 Exception 构造函数中形成错误消息。
 *
 * @param[in] fmt 与 `printf` 兼容的格式化说明符。
 * @details
 * | 类型 | 限定符 |
 * | ---- | ------ |
 * |`const char*`|`%s`|
 * |`char`|`%c`|
 * |`float` or `double`|`%f`,`%g`|
 * |`int`, `long`, `long long`|`%d`, `%ld`, ``%lld`|
 * |`unsigned`, `unsigned long`, `unsigned long long`|`%u`, `%lu`, `%llu`|
 * |`uint64` \f$\to\f$ `uintmax_t`, `int64` \f$\to\f$ `intmax_t`|`%ju`, `%jd`|
 * |`size_t`|`%zu`|
 */
std::string format(const char *fmt, ...);

/**
 * @brief 触发非法内存操作
 * @note 当调用该函数时，默认错误处理程序会发出一个硬件异常，这可以使调试更加方便
 */
inline void breakOnError()
{
    static volatile int *p = nullptr;
    *p = 0;
}

/**
 * @brief 该类封装了有关程序中发生的错误的所有或几乎所有必要信息。异常通常是通过 SRVL_Error
 *        和 SRVL_Error_ 宏隐式构造和抛出的 @see error
 */
class Exception final : public std::exception
{
public:
    //! @brief 默认构造
    Exception() : code(SRVL_StsOk), line(0) {}

    /**
     * @brief 完整的构造函数。通常不显式调用构造函数。而是使用宏 SRVL_Error()、SRVL_Error_()
     *        和 SRVL_Assert()
     */
    Exception(int _code, const std::string &_err, const std::string &_func, const std::string &_file, int _line);

    virtual ~Exception() noexcept = default;

    //! @return 错误描述和上下文作为文本字符串
    inline const char *what() const noexcept override { return msg.c_str(); }

    std::string msg;  //!< 格式化的错误信息
    int code;         //!< 错误码 @see SRVLErrorCode
    std::string err;  //!< 错误描述
    std::string func; //!< 函数名，仅在编译器支持获取时可用

    std::string file; //!< 发生错误的源文件名
    int line;         //!< 源文件中发生错误的行号
};

/**
 * @brief
 *
 * @param exc
 */
inline void throwError(const Exception &exc) { throw exc; }

#ifdef NDEBUG
#define SRVL_ERRHANDLE(exc) throwError(exc)
#else
#define SRVL_ERRHANDLE(...) breakOnError()
#endif

/**
 * @brief 发出错误信号并引发异常
 * @note 该函数将错误信息打印到stderr
 *
 * @param[in] _code 错误码
 * @param[in] _err 错误描述
 * @param[in] _func 函数名，仅在编译器支持获取时可用
 * @param[in] _file 发生错误的源文件名
 * @param[in] _line 源文件中发生错误的行号
 * @see SRVL_Error, SRVL_Error_, SRVL_Assert
 */
void error(int _code, const std::string &_err, const char *_func, const char *_file, int _line);

/**
 * @brief 返回完整的配置输出，返回值是原始的 CMake 输出，包括版本控制系统修订，编译器版本，编译器标志，启用的模块和第三方库等。
 *
 * @return const&
 */
const std::string &getBuildInformation();

} // namespace srvl

/**
 * @brief 调用错误处理程序
 * @note 目前，错误处理程序将错误代码和错误消息打印到标准错误流 `stderr`。在 Debug
 *       配置中，它会引发内存访问冲突，以便调试器可以分析执行堆栈和所有参数。在
 *       Release 配置中，抛出异常。
 *
 * @param[in] code 一种 SRVLErrorCode 错误码
 * @param[in] msg 错误信息
 */
#define SRVL_Error(code, msg) srvl::error(code, msg, __func__, __FILE__, __LINE__)

/**
 * @brief 调用错误处理程序
 * @note 该宏可用于动态构造错误消息，以包含一些动态信息，例如
 * @code
 *  // 请注意格式化文本消息周围的额外括号
 *  SRVL_Error_(StsOutOfRange, "the value at (%d, %d) = %g is out of range", badPt.x, badPt.y, badValue);
 * @endcode
 * @param[in] code 一种 SRVLErrorCode 错误码
 * @param[in] args 括号中带有类似 printf 格式的错误信息
 */
#define SRVL_Error_(code, fmt, args...) srvl::error(code, srvl::format(fmt, args), __func__, __FILE__, __LINE__)

/**
 * @brief 在运行时检查条件，如果失败则抛出异常
 * @note 宏 SRVL_Assert 对指定的表达式求值。如果它是0，宏抛出一个错误。
 * @see SRVLErrorCode
 */
#define SRVL_Assert(expr) (!!(expr)) ? (void(0)) : srvl::error(SRVL_StsAssert, #expr, __func__, __FILE__, __LINE__)

//! @} core
