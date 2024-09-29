# 纯碎碎念

#### 1.

target_link_libraries() can link the libraries in the deeper folders directly( target_link_libraries(sever A1 A2 M1 M2 kalman math) )

#### 2.

use the interface libraries(see kalman)

#### 3.

if you use target_include_directories() in the CMakeLists that builds the library, then when you link it, you're unnecessary to include_directories(). However, it won't cause a error.

#### 4.

1. 包含源文件的子文件夹**包含**CMakeLists.txt 文件，主目录的 CMakeLists.txt 通过 add_subdirectory 添加子目录即可；
2. 包含源文件的子文件夹**未包含**CMakeLists.txt 文件，子目录编译规则体现在主目录的 CMakeLists.txt 中；

#### 5.

changed:/home/yoda/25-Vision-LiuXiang/CMakeSkeletonBuilding/CMake_I/modules/A2/include/A2.h **add** #include<stddef>
