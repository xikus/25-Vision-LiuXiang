# CMake 考核任务

### 1. 总要求

使用 `CMakeLists.txt` 完成 C++ 代码的编译构建，补全缺少的 `CMakeLists.txt` 文件以及已提供 `CMakeLists.txt` 文件中的代码，使得整个项目能够正常构建。

### 2. 构建要求

#### 2.1 要求的编译选项

- `BUILD_TESTS`：构建测试（默认开启），能通过选择开启此编译选项来打开单元测试

- `BUILD_EXAMPLES`：构建示例程序（默认开启），能通过选择开启此编译选项来启用位于 samples 文件夹下的示例程序的构建

#### 2.2 构建命令

项目可通过以下命令行完成构建

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### 3. 运行要求

#### 3.1 单元测试

```shell
ctest
```

能显示 8 个测试用例运行成功

```txt
    Start  1: ArrayToolTest.linear2D
1/8 Test  #1: ArrayToolTest.linear2D .......................................   Passed    x.xx sec
    Start  2: ArrayToolTest.circular2D
2/8 Test  #2: ArrayToolTest.circular2D .....................................   Passed    x.xx sec
    Start  3: ArrayToolTest.linear3D
3/8 Test  #3: ArrayToolTest.linear3D .......................................   Passed    x.xx sec
    Start  4: ArrayToolTest.circular3D
4/8 Test  #4: ArrayToolTest.circular3D .....................................   Passed    x.xx sec
    Start  5: PretreatTest.null_image_input
5/8 Test  #5: PretreatTest.null_image_input ................................   Passed    x.xx sec
    Start  6: PretreatTest.1_channel_brightness
6/8 Test  #6: PretreatTest.1_channel_brightness ............................   Passed    x.xx sec
    Start  7: PretreatTest.3_channel_brightness
7/8 Test  #7: PretreatTest.3_channel_brightness ............................   Passed    x.xx sec
    Start  8: PretreatTest.3_channel_minus
8/8 Test  #8: PretreatTest.3_channel_minus .................................   Passed    x.xx sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =   x.xx sec
```

#### 3.2 示例程序

```shell
./srvl_version
```

这是在开启 `BUILD_EXAMPLES` 编译选项后构建生成的可执行文件，用于收集版本控制的信息，显示对应的版本号，运行结果如下。

```txt
3.6.0-dev
```

### 4. 备注与说明

#### 4.1 帮助

所有的 CMake 知识点在 `CMake.tar.xz` 的压缩包中均有指出，如果仍有不清楚的地方，请查阅 [CMake 官方文档](https://cmake.org/cmake/help/latest/)。

#### 4.2 备注

项目中除了给定的 `CMakeLists.txt` 可以添加语句外，其余文件**禁止**修改，包括 `*.h`、`*.hpp`、`*.cpp`、`*.cmake`、`*.in` 文件。

#### 4.3 如何构建测试

- 在 `cmake/SRVLCompilerOptions.cmake` 文件中定义了能自动获取 GoogleTest 单元测试工具的脚本，需要在包含该文件之前添加相关的编译选项。包含该文件后会自动下载 GoogleTest 并参与项目的构建，无需人为干涉。

- 在使用上与正常的目标构建完全一致，仅需要额外链接一个 `GTest::gtest_main` 目标即可，无需增加头文件依赖，无需增加可执行文件构建配置。

#### 4.4 参数构建

可以留意到在 `modules/ml` 文件夹下有一个 param 文件夹，因此其在构建中也需要涉及到关于参数构建的内容，因此需要在项目构建前包含有关参数构建的 `para_generator.cmake` 文件，并在 `ml` 模块的 `CMakeLists.txt` 中使用 `para_generator.cmake` 文件中定义的宏来实现参数 C++ 文件的生成。

##### 提示

* 参数规范文件解析与 C++ 文件的生成可参考 `para_generator.cmake` 文件，内部有示例以及使用说明。

* 对于 `ort` 目标而言，其模块名称为 `ml`。

* 如果想生成一个模块的 C++ 参数文件，可考虑使用 `srvl_generate_module_para`。
