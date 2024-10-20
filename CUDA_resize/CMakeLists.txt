cmake_minimum_required(VERSION 3.10)
project(image_resize_cuda_proj LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)   

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(CUDA REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(processor)

add_executable(test main.cpp)

target_link_libraries(test PUBLIC processor)

