/**
 * @file core_c.cpp
 * @author zhaoxi (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2023-04-21
 *
 * @copyright Copyright 2023 (c), zhaoxi
 *
 */

#include <cstdarg>

#include "srvl/core/util.hpp"

const char *srvlErrorStr(SRVLErrorCode status)
{
    static char buf[256];
    switch (status)
    {
    case SRVL_StsOk:
        return "No Error";
    case SRVL_StsBackTrace:
        return "Backtrace";
    case SRVL_StsError:
        return "Unspecified error";
    case SRVL_StsNoMem:
        return "Insufficient memory";
    case SRVL_StsBadArg:
        return "Bad argument";
    case SRVL_StsBadSize:
        return "Incorrect size of the array";
    case SRVL_StsNullPtr:
        return "Null pointer";
    case SRVL_StsDivByZero:
        return "Division by zero occurred";
    case SRVL_StsOutOfRange:
        return "One of the arguments\' values is out of range";
    case SRVL_StsAssert:
        return "Assertion failed";
    case SRVL_StsInvFmt:
        return "Invalid format";
    case SRVL_BadDynamicType:
        return "Bad dynamic_cast type";
    }

    sprintf(buf, "Unknown %s code %d", status >= 0 ? "status" : "error", status);
    return buf;
}

std::string srvl::format(const char *fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);
    std::string str(buf);
    return str;
}

srvl::Exception::Exception(SRVLErrorCode _code, const std::string &_err, const std::string &_func,
                           const std::string &_file, int _line)
    : code(_code), err(_err), func(_func), file(_file), line(_line)
{
    if (!func.empty())
        msg = srvl::format("SRVL %s: %d: \033[31;1merror\033[0m: (%d:%s) in function \"%s\"\n\033[34m"
                           ">>>>>>>> message >>>>>>>>\033[0m\n%s\n\033[34m<<<<<<<< message <<<<<<<<\033[0m\n",
                           file.c_str(), line, code, srvlErrorStr(code), func.c_str(), err.c_str());
    else
        msg = srvl::format("SRVL %s: %d: \033[31;1merror\033[0m: (%d:%s)\n\033[34m"
                           ">>>>>>>> message >>>>>>>>\033[0m\n%s\n\033[34m<<<<<<<< message <<<<<<<<\033[0m\n",
                           file.c_str(), line, code, srvlErrorStr(code), err.c_str());
}

void srvl::error(int _code, const std::string &_err, const char *_func, const char *_file, int _line)
{
    srvl::Exception exc(_code, _err, _func, _file, _line);

    SRVL_ERRHANDLE(exc);

#ifdef __GNUC__
#if !defined __clang__ && !defined __APPLE__
    // this suppresses this warning: "noreturn" function does return [enabled by default]
    __builtin_trap();
    // or use infinite loop: for (;;) {}
#endif
#endif //! __GNUC__
}

const std::string &srvl::getBuildInformation()
{
    static std::string build_info =
#include "version_string.inc"
        ;
    return build_info;
}
