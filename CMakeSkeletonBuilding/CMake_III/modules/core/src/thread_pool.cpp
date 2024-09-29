/**
 * @file thread_pool.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 非优先型线程池
 * @version 1.0
 * @date 2022-01-20
 *
 * @copyright Copyright SCUT RobotLab(c) 2021
 *
 */

#include "srvl/core/thread_pool.hpp"

using namespace std;

ThreadPool::~ThreadPool()
{
    if (__is_start)
        stop();
}

void ThreadPool::initCapacity(int capacity)
{
    // 初始化必须在未启动之前
    SRVL_Assert(!__is_start);
    __init_thread_size = capacity;
}

void ThreadPool::start()
{
    SRVL_Assert(__threads.empty());
    __is_start = true;
    __threads.reserve(__init_thread_size);
    for (int i = 0; i < __init_thread_size; i++)
        __threads.emplace_back(make_unique<thread>(
            [this]
            {
                loop();
            }));
}

void ThreadPool::join()
{
    while (!__tasks.empty() || __running_size > 0)
        ;
}

void ThreadPool::stop()
{
    {
        unique_lock<mutex> lk(__mutex);
        __is_start = false;
        __cond.notify_all();
    }

    for (auto &thread : __threads)
    {
        thread->join();
        thread = nullptr;
    }
    __threads.clear();
}

void ThreadPool::loop()
{
#ifdef THREAD_MESSAGE
    cout << "loop() tid : " << this_thread::get_id() << " start." << endl;
#endif
    while (__is_start)
    {
        Task task = takeTask();
        if (task)
        {
            task();
            unique_lock<mutex> lk(__mutex);
            --__running_size;
        }
    }
#ifdef THREAD_MESSAGE
    cout << "loop() tid : " << this_thread::get_id() << " exit." << endl;
#endif
}

Task ThreadPool::takeTask()
{
    // 上锁，锁包装体对象销毁时解锁
    unique_lock<mutex> lk(__mutex);
#ifdef THREAD_MESSAGE
    cout << "take() tid : " << this_thread::get_id() << " wait." << endl;
#endif
    // 任务队列不为空，或线程池未启动，线程不受条件变量控制
    // 启动后若任务队列为空，执行后线程阻塞，等待 addTask() 发出信号量后唤醒
    __cond.wait(lk,
                [&]() -> bool
                {
                    return !__tasks.empty() || !__is_start;
                });
    // 唤醒后上锁
#ifdef THREAD_MESSAGE
    cout << "take() tid : " << this_thread::get_id() << " wait up." << endl;
#endif
    [[maybe_unused]] int size = static_cast<int>(__tasks.size());
    // 从队列中获取队尾任务
    Task task;
    if (!__tasks.empty() && __is_start)
    {
        task = __tasks.front();
        __tasks.pop();
        SRVL_Assert(static_cast<int>(__tasks.size()) == (size - 1));
        ++__running_size;
    }
    return task;
}

/**
 * @brief Destroy the PriorityThreadPool object
 */
PriorityThreadPool::~PriorityThreadPool()
{
    if (__is_start)
    {
        stop();
    }
}

/**
 * @brief 初始化线程池容量
 *
 * @param capacity 容量
 */
void PriorityThreadPool::initCapacity(int capacity)
{
    // 初始化必须在未启动之前
    SRVL_Assert(!__is_start);
    __init_thread_size = capacity;
}

/**
 * @brief 启用线程池
 */
void PriorityThreadPool::start()
{
    SRVL_Assert(__threads.empty());
    __is_start = true;
    __threads.reserve(__init_thread_size);
    for (int i = 0; i < __init_thread_size; i++)
    {
        __threads.emplace_back(make_unique<thread>(
            [this]
            {
                loop();
            }));
    }
}

/**
 * @brief 执行完毕所有任务后汇合主线程
 */
void PriorityThreadPool::join()
{
    while (!__tasks.empty() && __running_size > 0)
        ;
}

/**
 * @brief 主线程阻塞至所有线程运行完毕，停用线程池，清空任务队列，
 *        此时可被重新初始化线程大小
 */
void PriorityThreadPool::stop()
{
    {
        unique_lock<mutex> lk(__mutex);
        __is_start = false;
        __cond.notify_all();
    }

    for (auto &thread : __threads)
    {
        thread->join();
        thread = nullptr;
    }
    __threads.clear();
}

/**
 * @brief 循环执行取用的任务
 */
void PriorityThreadPool::loop()
{
#ifdef THREAD_MESSAGE
    cout << "loop() tid : " << this_thread::get_id() << " start." << endl;
#endif
    while (__is_start)
    {
        Task task = takeTask();
        if (task)
        {
            task();
            unique_lock<mutex> lk(__mutex);
            --__running_size;
        }
    }
#ifdef THREAD_MESSAGE
    cout << "loop() tid : " << this_thread::get_id() << " exit." << endl;
#endif
}

/**
 * @brief 从任务队列中取用任务
 *
 * @return 取用到的任务
 */
Task PriorityThreadPool::takeTask()
{
    unique_lock<mutex> lk(__mutex);
#ifdef THREAD_MESSAGE
    // 任务队列为空，或线程池未启动则将 notify() 作为虚假唤醒
    cout << "take() tid : " << this_thread::get_id() << " wait." << endl;
#endif
    __cond.wait(lk,
                [&]() -> bool
                {
                    return !__tasks.empty() || !__is_start;
                });
#ifdef THREAD_MESSAGE
    cout << "take() tid : " << this_thread::get_id() << " wait up." << endl;
#endif
    [[maybe_unused]] int size = static_cast<int>(__tasks.size());
    // 从大顶堆 (优先队列) 中获取堆顶任务
    Task task;
    if (!__tasks.empty() && __is_start)
    {
        task = __tasks.top().first;
        __tasks.pop();
        SRVL_Assert(static_cast<int>(__tasks.size()) == (size - 1));
        ++__running_size;
    }
    return task;
}
