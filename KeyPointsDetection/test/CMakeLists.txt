cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(test LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")



set(TensorRT_LIBRARY /home/yoda/TensorRT-8.6.1.6/lib)
set(TensorRT_INCLUDE_DIR /home/yoda/TensorRT-8.6.1.6/include)
set(TensorRT_LIB_DIR /home/yoda/TensorRT-8.6.1.6/lib)


# CUDA
find_package(CUDA REQUIRED)

get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)

# OpenCV
find_package(OpenCV REQUIRED)


# TensorRT




list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)

list(APPEND ALL_LIBS
        
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer.so.8.6.1
        

)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)
# link_libraries(/home/yoda/TensorRT-8.6.1.6/lib/libnvinfer.so.8.6.1)


include_directories(/home/yoda/TensorRT-8.6.1.6/include)


add_executable(${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_INCLUDE_DIRS}
)

target_link_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_LIB_DIRS}
        /home/yoda/TensorRT-8.6.1.6/lib

)

target_link_libraries(
        ${PROJECT_NAME}
        PUBLIC
        ${ALL_LIBS}
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvparsers.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer_builder_resource.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer_dispatch.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer_lean.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvinfer_vc_plugin.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvonnxparser.so.8.6.1
        /home/yoda/TensorRT-8.6.1.6/lib/libnvparsers.so.8.6.1
        
)


if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif ()
