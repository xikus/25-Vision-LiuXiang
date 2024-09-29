/**
 * @file timer.hpp
 * @author 范文宇 (2643660853@qq.com)
 * @brief 
 * @version 1.0
 * @date 2023-01-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <string>
#include <chrono>
#include <thread>
#include <functional>

//! @addtogroup core
//! @{
//! @defgroup timer 定时、计时模块
//! @}

//! @addtogroup timer
//! @{

/**
 * @brief 获取当前系统时间
 * @note 当参数是 std::string 时，返回 yyyy-mm-dd hh:mm:sss 的 std::string 类型
 *       当参数是 int64_t 时，返回 Linux 发行时间到当前的毫秒数
 * @tparam _Tp 可选类型，std::string 或 int64_t
 * @return 系统时间
 */
template <typename _Tp,
          typename Enable = typename std::enable_if<std::is_same<_Tp, std::string>::value ||
                                                    std::is_same<_Tp, int64_t>::value>::type>
_Tp getSystemTick();

/**
 * @brief 计时器
 */
class Timer
{
    bool __started;                                    //!< 开始计时标志位
    bool __paused;                                     //!< 停止计时标志位
    std::chrono::steady_clock::time_point __reference; //!< 当前时间
    std::chrono::duration<long double> __accumulated;  //!< 累计时间

public:
    /**
     * @brief 创建计时器对象
     */
    Timer() : __started(false), __paused(false),
              __reference(std::chrono::steady_clock::now()),
              __accumulated(std::chrono::duration<long double>(0)) {}

    Timer(const Timer &) = delete;
    Timer(Timer &&) = delete;
    ~Timer() = default;

    /**
     * @brief 开始计时
     */
    void start();

    /**
     * @brief 结束计时
     */
    void stop();

    /**
     * @brief 重置计时器
     */
    void reset();

    /**
     * @brief 计算计时时长
     * @note 计算 start() 和 stop() 函数之间的时长
     *
     * @return 计时时长，单位是 ms
     */
    std::chrono::milliseconds::rep duration() const;
};

/**
 * @brief 同步定时，当前进程挂起time_length时长后，再继续当前进程
 *
 * @param[in] time_length 定时时长
 */
void syncWait(int64_t time_length);

/**
 * @brief 异步定时，另外开一个线程，在定时time_length ms后执行某个函数
 *
 * @tparam _Callable 可调用对象
 * @tparam _Args 可调用参数
 * @param[in] time_length 定时时长
 * @param[in] f 函数指针
 * @param[in] args 函数的参数
 */
template <typename _Callable, class... _Args>
void asyncWait(int after, _Callable &&f, _Args &&...args)
{
    // 函数包装器包装输入的函数指针和函数参数
    std::function<typename std::result_of<_Callable(_Args...)>::type()> task(std::bind(std::forward<_Callable>(f), std::forward<_Args>(args)...));
    // 另外开一个线程进行定时
    std::thread(
        [after, task]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(after));
            task();
        })
        .detach();
}

//! @} timer
