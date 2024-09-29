/**
 * @file singleton.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 单例类模板
 * @version 1.0
 * @date 2022-02-08
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <cassert>
#include <utility>

//! @addtogroup core
//! @{

/**
 * @brief 单例模板
 *
 * @tparam _Tp 要更改为单例的模板参数的类型
 */
template <typename _Tp>
class GlobalSingleton final
{
public:
    //! 获取实例化对象
    static _Tp *Get() { return *GetPPtr(); }

    //! 构造单例
    template <typename... Args>
    static void New(Args &&...args)
    {
        assert(Get() == nullptr);
        *GetPPtr() = new _Tp(std::forward<Args>(args)...);
    }

    //! 销毁单例
    static void Delete()
    {
        if (Get() != nullptr)
        {
            delete Get();
            *GetPPtr() = nullptr;
        }
    }

private:
    //! 获取二级指针
    static _Tp **GetPPtr()
    {
        static _Tp *ptr = nullptr;
        return &ptr;
    }
};

//! @} core
