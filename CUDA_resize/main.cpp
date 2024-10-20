#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <chrono>
#include <processer.h>
#include <processer_cuda_fun.cuh>
int main() {

    int inputWidth = 1920;
    int inputHeight = 1080;
    int outputWidth = 640;
    int outputHeight = 640;

    // 读取图片
    const char* image_path = "../test.jpg";
    uchar* h_inputImage = read_image(image_path);

    // malloc host 
    uchar* h_outputImage = new unsigned char[outputWidth * outputHeight * 3];

    // 调用cuda launch函数
    bilinearInterpolation_launch((uchar3*)h_inputImage, (uchar3*)h_outputImage, inputWidth, inputHeight, outputWidth, outputHeight);

    // save img 
    const char* output_filename = "./output.jpg";
    stbi_write_png(output_filename, outputWidth, outputHeight, 3, h_outputImage, outputWidth * 3);

    // free cpu 
    delete[] h_inputImage;
    delete[] h_outputImage;

    return 0;
}