/**
 * @file sample_version.cpp
 * @author zhaoxi (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2023-06-25
 *
 * @copyright Copyright 2023 (c), zhaoxi
 *
 */

#include <iostream>

#include <opencv2/core.hpp>

#include "srvl/core.hpp"

const char *keys = "{ help h usage ? |  | 帮助信息 }";

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
        parser.printMessage();
    else
        std::cout << SRVL_VERSION << std::endl;
    return 0;
}
