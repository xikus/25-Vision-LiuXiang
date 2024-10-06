![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

# 华南虎视觉组实习生任务 CMake

**实习生姓名：柳翔**

> **问题 1:** 源文件缺少`#include <header>`
> **解决 1:** 在 `CMakeLists.txt` 使用`target_compile_options(TARGET_NAME PUBLIC -include <header>)`
> **解决 2:** 在源文件中加入`#include <header>`

---

> **问题 2:** 在一个主 `CMakeLists.txt`下,链接库时不用考虑库是否在子目录或者父目录中,只要库在主目录的范围内, 都可以`target_link_libraries`或者`target_include_directories`

---

> **问题 3:** 对于库来说,在`add_library()`时如果使用`target_include_directories()`,且参数为`PUBLIC`,则在`target_link_libraries()`这个`lib`时无需再`target_link_directories()`,因为在编库时已经把库的路径包含了

---

> **问题 4:** `add_subdirectory()`写在`target_link_directories()`和`target_link_libraries()`之前. 因为只有添加了子目录以后才能找到子目录的`CMakeLists.txt`,然后才能链接其中的库

---

> **问题 5:** 源文件中有`#if`与`#endif`包含的内容

```
#ifdef WITH_A
void print()
{
    module1a m;
    m.print1(), m.print2(), m.print3(), m.print4();
}
#endif // WITH_A

#ifdef WITH_B
void print()
{
    module1b m;
    m.print();
}
#endif // WITH_B
```

> **解决:** 使用`add_definitions`可以为程序添加宏

```if(BUILD_A)
add_definitions(-DWITH_A)
endif()

if(BUILD_B)
add_definitions(-DWITH_B)
endif()
```

## CMake 结构

```
CMake_I
├── CMakeLists.txt
├── common
│   ├── CMakeLists.txt
│   ├── kalman
│   │   ├── CMakeLists.txt
│   │   └── include
│   │       └── KalmanFilterX.hpp
│   └── math
│       ├── CMakeLists.txt
│       ├── include
│       │   └── Math.h
│       └── src
│           └── Math.cpp
├── main.cpp
├── modules
│   ├── A1
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── A1.h
│   │   └── src
│   │       ├── A11.cpp
│   │       ├── A12.cpp
│   │       └── A13.cpp
│   ├── A2
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── A2.h
│   │   └── src
│   │       └── A2.cpp
│   ├── CMakeLists.txt
│   ├── M1
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── M1.h
│   │   └── src
│   │       └── M1.cpp
│   └── M2
│       ├── CMakeLists.txt
│       ├── include
│       │   └── M2.h
│       └── src
│           └── M2.cpp
└── README.md
```

```
CMake_II
├── client.cpp
├── CMakeLists.txt
├── common
│   ├── CMakeLists.txt
│   ├── rmath
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── robotlab
│   │   │       └── rmath.h
│   │   └── src
│   │       └── rmath.cpp
│   └── singleton
│       ├── CMakeLists.txt
│       └── include
│           └── robotlab
│               └── singleton.hpp
├── modules
│   ├── assembly1
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── robotlab
│   │   │       └── assembly1.h
│   │   ├── src
│   │   │   └── assembly1.cpp
│   │   └── test
│   │       ├── assembly1_test.cpp
│   │       └── CMakeLists.txt
│   ├── assembly2
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── robotlab
│   │   │       └── assembly2.h
│   │   └── src
│   │       └── assembly2.cpp
│   ├── CMakeLists.txt
│   ├── module1
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── robotlab
│   │   │       ├── module1
│   │   │       │   ├── module1a.h
│   │   │       │   └── module1b.h
│   │   │       └── module1.hpp
│   │   └── src
│   │       ├── module1a
│   │       │   ├── m1.cpp
│   │       │   ├── m2.cpp
│   │       │   ├── m3.cpp
│   │       │   └── m4.cpp
│   │       └── module1b
│   │           └── module1b.cpp
│   └── module2
│       ├── CMakeLists.txt
│       ├── include
│       │   └── robotlab
│       │       ├── opcua_cs
│       │       │   ├── client.hpp
│       │       │   ├── server.hpp
│       │       │   └── ua_type
│       │       │       ├── argument.hpp
│       │       │       ├── object.hpp
│       │       │       ├── open62541.h
│       │       │       ├── ua_define.hpp
│       │       │       └── variable.hpp
│       │       └── opcua_cs.hpp
│       └── src
│           ├── argument.cpp
│           ├── client.cpp
│           ├── object.cpp
│           ├── open62541.c
│           ├── server.cpp
│           └── variable.cpp
├── README.md
└── server.cpp
```
