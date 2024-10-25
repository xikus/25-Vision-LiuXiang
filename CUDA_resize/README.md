![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

### CUDA编程

- **文件结构**
>```bash
>CUDA_resize
>├── CMakeLists.txt
>├── compare
>├── compare.cpp
>├── main.cpp
>├── output.jpg
>├── processor
>│   ├── CMakeLists.txt
>│   ├── include
>│   │   ├── processer_cuda_fun.cuh
>│   │   └── processer.h
>│   └── src
>│       ├── processer_cuda_fun.cu
>│       └── processor.cu
>├── README.md
>└── test.jpg
>```
>`processor_cuda_fun`内有`bilinearInterpolation_test`,利用双线性插值来计算`output`每个像素rgb. `bilinearInterpolation_test`的核心函数是`bilinearInterpolationKernel`, 负责并行计算.
>`processor`内有`read_image`, `bilinearInterpolation_launch`等更高级的函数.

- **运行**
>```bash
>mkdir build
>cd build
>cmake ..
>make
>./test
>```

- **运行效果**
>```bash
>Image loaded: ../test.jpg
>Width: 1920 Height: 1080 Channels: 3
>sizeof(uchar3) = 3
>allocating memory duration: 1628.13 ms
>h2d status = 0
>blockSize: x =16,y = 16,z =1
>gridSize: x = 40,y=40,z = 1
>kernel duration: 0.138195 ms
>```
**对照组(OpenCV):用时1ms**

- **遇到的问题**
> target_link_directories(${CUDA_LIBRARIES})无果,CUDA_LIBRARIES打印后发现残缺
> ***
> 解决:
>find_package(CUDA REQUIRED)
>
>include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
>
>link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

#### 总结
**CUDA编程对于入门的我来说重点在于三方面: 并行计算, Host与Device之间的通信以及内存管理.**

- 并行化需要我们将一个函数分解为一个个相同的操作(不同的话会导致线程束分化,降低并行度), 将串行的操作转换为并行.如Resize()中对于每个像素的操作可以并行化.

- Host-Device通信对我来说是CUDA编程中比较难理解的部分. 为了避免通信的延时, cpu和gpu的数据传输都是异步进行的. 因此在并行运算完成后需要使用`cudaDeviceSynchronize()`来同步.

- 在gpu上需要分配input和output的内存,在运算完成后还要将gpu上的output给copy到cpu上.最后也不要忘了`cudafree`. 这些操作对于当时刚开始接触cuda的我还是有点难记的.