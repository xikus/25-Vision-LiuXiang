/**
 * @file thread_pool.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 非优先型线程池头文件
 * @version 1.0
 * @date 2022-01-20
 *
 * @copyright Copyright SCUT RobotLab(c) 2021
 *
 */

#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

#include "util.hpp"

//! @addtogroup core
//! @{
//! @defgroup thread_pool 基于 std::thread 的线程池
//! @}

//! @addtogroup thread_pool
//! @{

//! 任务优先级
enum class TaskPriority
{
    level_0, //!< 最高优先级
    level_1, //!< 中等优先级
    level_2  //!< 最低优先级
};

//! 任务
using Task = std::function<void()>;
//! 任务优先级对组
using TaskPair = std::pair<Task, TaskPriority>;

struct _Cmp
{
    bool operator()(TaskPair &p1, TaskPair &p2)
    {
        return p1.second > p2.second;
    }
};

//! 线程池
using Threads = std::vector<std::unique_ptr<std::thread>>;
//! 任务队列
using Tasks = std::queue<Task>;
//! 任务优先级队列
using TaskPairs = std::priority_queue<TaskPair, std::vector<TaskPair>, _Cmp>;

/**
 * @brief 非优先型线程池
 * @note 任务队列无优先级之分，使用 `std::queue<>` 储存，添加任务效率高
 */
class ThreadPool
{
private:
    bool __is_start = false;        //!< 线程池是否启动
    int __init_thread_size = 1;     //!< 初始化线程数量
    int __running_size = 0;         //!< 正在执行任务的线程
    Threads __threads;              //!< 线程池
    Tasks __tasks;                  //!< 任务优先队列
    std::mutex __mutex;             //!< 互斥锁
    std::condition_variable __cond; //!< 条件变量

public:
    //! 构造线程池
    ThreadPool() = default;

    /**
     * @brief 构造线程池
     *
     * @param[in] capacity 线程池的线程容量
     */
    explicit ThreadPool(int capacity) : __init_thread_size(capacity) {}
    
    //! 销毁线程池
    ~ThreadPool();
    
    //! 启用线程池，根据初始化线程池大小创建相应数量的线程
    void start();

    /**
     * @brief 初始化线程池容量
     *
     * @param[in] capacity 容量
     */
    void initCapacity(int capacity);

    //! 执行完毕所有任务后汇合主线程
    void join();

    //! 主线程阻塞至所有线程运行完毕，停用线程池，清空任务队列，此时可被重新初始化线程大小
    void stop();

    /**
     * @brief 添加任务，互斥锁 mutex 解锁 情况一：线程闲置时等待、unlock，线程唤醒后继续上锁。
     *        情况二：任务获取完毕、unlock。解锁后可唤醒 addTask() 所在线程（一般是主线程）
     *        并继续添加任务至队列
     *
     * @param[in] task 任务
     */
    template <typename _Task>
    void addTask(_Task &&task)
    {
        // 获取互斥锁，未加锁则继续运行
        std::unique_lock<std::mutex> lk(__mutex);
        __tasks.emplace(std::forward<_Task>(task));
        __cond.notify_one();
    }

    //! 获取正在执行任务的线程数
    inline int getRunningSize() { return __running_size; }

private:
    //! 循环执行取用的任务
    void loop();

    /**
     * @brief 从任务队列中取用任务，任务队列不为空，则此线程不等待信号量，addTask() 信号量作为虚假唤醒；
     *        任务队列为空，则此线程处于闲置状态，等待 addTask() 发来信号后唤醒
     *
     * @return 取用到的任务
     */
    Task takeTask();
};

/**
 * @brief 优先型线程池
 * @note 任务队列会按照优先级进行排序，使用 `std::priority_queue<>` 储存
 */
class PriorityThreadPool
{
private:
    bool __is_start = false;        //!< 线程池是否启动
    int __init_thread_size = 1;     //!< 初始化线程数量
    int __running_size = 0;         //!< 正在执行任务的线程
    Threads __threads;              //!< 线程池
    TaskPairs __tasks;              //!< 任务优先队列
    std::mutex __mutex;             //!< 互斥锁
    std::condition_variable __cond; //!< 条件变量

public:
    //! 构造线程池
    PriorityThreadPool() = default;

    /**
     * @brief 构造线程池
     *
     * @param[in] capacity 线程池的线程容量
     */
    explicit PriorityThreadPool(int capacity) : __init_thread_size(capacity) {}

    //! 销毁线程池
    ~PriorityThreadPool();

    //! 启用线程池，根据初始化线程池大小创建相应数量的线程
    void start();

    /**
     * @brief 初始化线程池容量
     *
     * @param[in] capacity 容量
     */
    void initCapacity(int);

    //! 执行完毕所有任务后汇合主线程
    void join();

    //! 主线程阻塞至所有线程运行完毕，停用线程池，清空任务队列，此时可被重新初始化线程大小
    void stop();

    /**
     * @brief 加入任务
     *
     * @param[in] task 任务
     * @param[in] priority 任务优先级
     */
    template <typename _Task>
    void addTask(_Task &&task, TaskPriority priority)
    {
        // 默认最低优先级
        std::unique_lock<std::mutex> lk(__mutex);
        __tasks.emplace(std::forward<_Task>(task), priority);
        __cond.notify_one();
    }

private:
    //! 循环执行取用的任务
    void loop();

    //! 从任务队列中取用任务
    Task takeTask();
};

//! @} thread_pool
