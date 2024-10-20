#include <cuda_runtime.h>
#include <iostream>
#include "processer.h"
#include "processer_cuda_fun.cuh"
#include <chrono>

// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
typedef unsigned char uchar;

// cpu 读取图像文件
unsigned char* read_image(const char* filename) {

    int width, height, channels;
    // 读取图像文件
    stbi_uc* imageData = stbi_load(filename, &width, &height, &channels, 0);
    if (imageData == nullptr) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
    }

    std::cout << "Image loaded: " << filename << std::endl;
    std::cout << "Width: " << width << " Height: " << height << " Channels: " << channels << std::endl;

    return imageData;
}

void bilinearInterpolation_launch(uchar3* h_inputImageUChar3,
    uchar3* h_outputImageUChar3,
    int inputWidth, int inputHeight,
    int outputWidth, int outputHeight) {
    uchar3* d_inputImage;
    uchar3* d_outputImage;

    size_t inputImageSize = inputWidth * inputHeight * sizeof(uchar3);
    size_t outputImageSize = outputWidth * outputHeight * sizeof(uchar3);
    cout << "sizeof(uchar3) = " << sizeof(uchar3) << endl;

    auto mem_start = std::chrono::high_resolution_clock::now();

    // cuda malloc && memset
    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);
    cudaMemset(d_inputImage, 0, inputImageSize);
    cudaMemset(d_outputImage, 0, outputImageSize);

    auto mem_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = mem_end - mem_start;
    std::cout << "allocating memory duration: " << duration.count() << " ms" << std::endl;


    // h2d
    auto status = cudaMemcpy(d_inputImage, h_inputImageUChar3, inputImageSize, cudaMemcpyHostToDevice);
    cout << "h2d status = " << status << endl;

    float scaleX = (float)(inputWidth - 1) / outputWidth;
    float scaleY = (float)(inputHeight - 1) / outputHeight;

    // cuda block/grid size
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, \
        (outputHeight + blockSize.y - 1) / blockSize.y, 1);
    cout << "blockSize: x =" << blockSize.x << ",y = " << blockSize.y << ",z =" << blockSize.z << endl;
    cout << "gridSize: x = " << gridSize.x << ",y=" << gridSize.y << ",z = " << gridSize.z << endl;

    auto kernel_start = std::chrono::high_resolution_clock::now();
    // 双线性插值算法
    bilinearInterpolationKernel<<<gridSize, blockSize >>>(d_inputImage, d_outputImage, inputWidth, inputHeight, outputWidth, outputHeight, scaleX, scaleY);

    auto kernel_end = std::chrono::high_resolution_clock::now();
    duration = kernel_end - kernel_start;
    std::cout << "kernel duration: " << duration.count() << " ms" << std::endl;

    // 同步设备
    cudaDeviceSynchronize();

    // 复制输出图像数据回主机
    cudaMemcpy(h_outputImageUChar3, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);

    
    // 释放设备内存
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    
}