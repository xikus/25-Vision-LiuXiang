[img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

# 华南虎视觉组实习生任务CMake

**实习生姓名：柳翔** 


> **问题1:** 源文件缺少`#include <header>` 
**解决1:** 在 `CMakeLists.txt` 使用```target_compile_options(TARGET_NAME PUBLIC -include <header>)```
**解决2:** 在源文件中加入`#include <header>`

** **
> **问题2:** 在一个主 `CMakeLists.txt`下,链接库时不用考虑库是否在子目录或者父目录中,只要库在主目录的范围内, 都可以`target_link_libraries`或者`target_include_directories`

** **
> **问题3:** 对于库来说,在`add_library()`时如果使用`target_include_directories()`,且参数为`PUBLIC`,则在`target_link_libraries()`这个`lib`时无需再`target_link_directories()`,因为在编库时已经把库的路径包含了

---
> **问题4:** `add_subdirectory()`写在`target_link_directories()`和`target_link_libraries()`之前. 因为只有添加了子目录以后才能找到子目录的`CMakeLists.txt`,然后才能链接其中的库

---
> **问题5:** 源文件中有`#if`与`#endif`包含的内容
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
## 总结