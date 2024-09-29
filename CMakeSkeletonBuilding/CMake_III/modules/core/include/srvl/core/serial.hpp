/**
 * @file serial.hpp
 * @author zhaoxi (535394140@qq.com)
 * @brief 
 * @version 1.0
 * @date 2022-11-22
 * 
 * @copyright Copyright SCUT RobotLab(c) 2022
 * 
 */

#pragma once

#pragma once

#include <memory>
#include <termios.h>
#include <vector>
#include <string>

//! @addtogroup core
//! @{
//! @defgroup serial 串口通信模块
//! @}

//! @addtogroup serial
//! @{

#ifdef NDEBUG
#define DEBUG_SER_WARNING(msg) ((void)0)
#define DEBUG_SER_ERROR(msg) ((void)0)
#define DEBUG_SER_HIGHLIGHT(msg) ((void)0)
#define DEBUG_SER_INFO(msg) ((void)0)
#define DEBUG_SER_PASS(msg) ((void)0)
#else
#define DEBUG_SER_WARNING(msg) SER_WARNING(msg)
#define DEBUG_SER_ERROR(msg) SER_ERROR(msg)
#define DEBUG_SER_HIGHLIGHT(msg) SER_HIGHLIGHT(msg)
#define DEBUG_SER_INFO(msg) SER_INFO(msg)
#define DEBUG_SER_PASS(msg) SER_PASS(msg)
#endif

#define SER_HIGHLIGHT(msg...)                \
    do                                       \
    {                                        \
        printf("\033[35m[ SER-INFO ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define SER_WARNING(msg...)                  \
    do                                       \
    {                                        \
        printf("\033[33m[ SER-ERR  ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define SER_PASS(msg...)                     \
    do                                       \
    {                                        \
        printf("\033[32m[ SER-PASS ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define SER_ERROR(msg...)                    \
    do                                       \
    {                                        \
        printf("\033[31m[ SER-ERR  ] " msg); \
        printf("\033[0m\n");                 \
    } while (false)

#define SER_INFO(msg...)             \
    do                               \
    {                                \
        printf("[ SER-INFO ] " msg); \
        printf("\n");                \
    } while (false)

/**
 * @brief Serial communication library
 */
class SerialPort
{
    int __fd;               //!< 文件描述符
    termios __option;       //!< 终端控制
    bool __is_open = false; //!< 串口打开标志位
    std::string __device;   //!< 设备名
    int __baud_rate;        //!< 波特率

public:
    /**
     * @brief Construct a new SerialPort object
     *
     * @param[in] device Device name, default is empty
     * @param[in] baud_rate Baud rate, default B115200
     */
    explicit SerialPort(const std::string &device = {}, int baud_rate = B115200)
        : __device(device), __baud_rate(baud_rate) { open(); }

    /**
     * @brief Destroy the SerialPort object
     */
    ~SerialPort() { close(); }

    /**
     * @brief Open the serial port, search all available devices automatically and try to open the first
     */
    void open();

    /**
     * @brief Close the serial port
     */
    void close();

    /**
     * @brief 从串口读取结构体
     *
     * @tparam T 读取到结构体的类型
     * @param[in] head 头帧
     * @param[in] tail 尾帧
     * @return std::vector<T> 读到的所有结构体
     */
    template <typename T>
    std::vector<T> readStruct(unsigned char head, unsigned char tail)
    {
        std::vector<T> vec_t;
        constexpr int LENGTH = 512;
        constexpr int SIZE = sizeof(T);
        unsigned char read_buffer[LENGTH] = {0};
        ssize_t len_result = read(read_buffer, LENGTH);
        for (ssize_t i = 0; (i + SIZE + 1) < len_result; i++)
            if (read_buffer[i] == head && read_buffer[i + SIZE + 1] == tail)
                vec_t.push_back(*(reinterpret_cast<T *>(&read_buffer[i + 1])));
        return vec_t;
    }

    /**
     * @brief Struct data write to serial port
     *
     * @tparam _Tp 写入结构体的类型
     * @param[in] data_struct 要写入的结构体
     * @return 是否写入成功
     */
    template <typename _Tp>
    bool writeStruct(_Tp &data_struct)
    {
        ssize_t len_result = write(&data_struct, sizeof(data_struct));
        return (sizeof(data_struct) == len_result);
    }

    /**
     * @brief The serial is opened?
     *
     * @return true
     * @return false
     */
    inline bool isOpened() const { return __is_open; };

private:
    /**
     * @brief Write data
     *
     * @param[in] data Start position of the data
     * @param[in] length The length of the data to be written
     * @return Whether to write in full
     */
    ssize_t write(void *data, size_t len);

    /**
     * @brief Read data
     *
     * @param[in] data Start position of the data
     * @param[in] len The length of the data to be read
     * @return Length
     */
    ssize_t read(void *data, size_t len);
};

using serial_ptr = std::unique_ptr<SerialPort>;

//! @} serial
