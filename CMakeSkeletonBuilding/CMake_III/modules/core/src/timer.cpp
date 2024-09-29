/**
 * @file timer_schedule.cpp
 * @author zhaoxi (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2023-01-30
 *
 * @copyright Copyright SCUT RobotLab(c) 2023
 *
 */

#include <sstream>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>

#include "srvl/core/timer.hpp"

using namespace std;

template <>
string getSystemTick<string>()
{
    // 获取实时时间
    chrono::system_clock::time_point now = chrono::system_clock::now();
    time_t now_time_t = chrono::system_clock::to_time_t(now);
    tm *now_tm = localtime(&now_time_t);
    // 将实时时间格式化
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%F %T", now_tm);
    // 为了添加毫秒数的输出，运用ostringstream
    ostringstream as;
    as.fill('0');
    chrono::milliseconds ms;
    ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
    as << buffer << ":" << ms.count();
    return as.str();
}

template <>
int64_t getSystemTick<int64_t>()
{
    chrono::system_clock::time_point now = chrono::system_clock::now();
    chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch());
    return static_cast<int64_t>(ms.count());
}

void Timer::start()
{
    // 增加标志位的判断，避免重复计时
    // 如果开始计时标志位为false
    if (!__started)
    {
        // 开始计时标志位设为true
        __started = true;
        // 结束计时标志位设为false
        __paused = false;
        // 计时积累时间重置为0
        __accumulated = chrono::duration<long double>(0);
        // 记录当前时间
        __reference = chrono::steady_clock::now();
    }
    // 如果结束计时标志位为true
    else if (__paused)
    {
        // 更新当前时间
        __reference = chrono::steady_clock::now();
        // 结束计时标志位设为false
        __paused = false;
    }
}

void Timer::stop()
{
    // 如果已经开始计时并且没有停止计时
    if (__started && !__paused)
    {
        // 记录当前时间
        chrono::steady_clock::time_point now = chrono::steady_clock::now();
        // 计算start到stop之间的时间差
        __accumulated = __accumulated + chrono::duration_cast<chrono::duration<long double>>(now - __reference);
        // 将停止计时标志位设为true
        __paused = true;
    }
}

void Timer::reset()
{
    // 重置计时器的开始标志位
    __started = false;
    // 重置计时器的暂停标志位
    __paused = false;
    // 更新当前时间
    __reference = chrono::steady_clock::now();
    // 将计时积累时间重置为0
    __accumulated = chrono::duration<long double>(0);
}

chrono::milliseconds::rep Timer::duration() const
{
    // 如果尚未开始计时，则返回0
    if (!__started)
        return chrono::milliseconds(0).count();
    // 如果是开始计时标志位和停止计时标志位都设置为true了
    if (__paused)
        // 返回累计计时时长
        return chrono::duration_cast<chrono::milliseconds>(__accumulated).count();
    // 如果已经开始计时，但是还没有结束计时，返回至今已累计的计时时长
    return chrono::duration_cast<chrono::milliseconds>(
               __accumulated + (chrono::steady_clock::now() - __reference))
        .count();
}

void syncWait(int64_t time_length)
{
    // 定义信号类型 SIGALRM
    if (signal(SIGALRM, [](int) {}) == SIG_ERR)
        perror("signal error\n");
    // 设定定时时长
    itimerval tv = {.it_interval = {.tv_sec = 0, .tv_usec = 0},
                    .it_value = {.tv_sec = 0, .tv_usec = 0}};

    time_length < 1e6
        ? tv.it_value.tv_usec = time_length
        : tv.it_value.tv_sec = time_length / 1e6,
          tv.it_value.tv_usec = time_length - tv.it_value.tv_sec * 1e6;

    // 定时时长不合理则直接编译不通过，并且再一定的时间后发送信号
    if (setitimer(ITIMER_REAL, &tv, NULL) != 0)
        perror("setitimer error\n");
    // 进程挂起，收到信号之后再继续当前进程
    pause();
}
