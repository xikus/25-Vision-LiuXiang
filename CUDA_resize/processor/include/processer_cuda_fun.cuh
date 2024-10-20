#ifndef PROCESSOR_CUDA_FUN_CUH
#define PROCESSOR_CUDA_FUN_CUH

typedef unsigned char uchar;

// 计算每个像素rgb的插值结果
__device__ uchar3  bilinearInterpolation_test(float srcX, float srcY,
    uchar3* d_inputImage, int inputWidth, int inputHeight);

__global__ void bilinearInterpolationKernel(
    uchar3* d_inputImage,
    uchar3* d_outputImage,
    int inputWidth, int inputHeight,
    int outputWidth, int outputHeight,
    float scaleX, float scaleY
);
#endif // PROCESSOR_CUDA_FUN_CUH