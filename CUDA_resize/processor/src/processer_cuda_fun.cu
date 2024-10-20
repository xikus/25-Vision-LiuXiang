
#include "processer_cuda_fun.cuh"

// 计算每个像素rgb的插值结果
__device__ uchar3  bilinearInterpolation_test(float srcX, float srcY,
    uchar3* d_inputImage, int inputWidth, int inputHeight) {
    // 找到周围的四个像素
    int x1 = (int)floor(srcX);
    int y1 = (int)floor(srcY);
    int x2 = min(x1 + 1, inputWidth - 1);
    int y2 = min(y1 + 1, inputHeight - 1);

    // 计算插值权重
    float wx = srcX - x1;
    float wy = srcY - y1;

    // 双线性插值计算（相邻四个点的像素值）
    uchar3 p1 = d_inputImage[y1 * inputWidth + x1];
    uchar3 p2 = d_inputImage[y1 * inputWidth + x2];
    uchar3 p3 = d_inputImage[y2 * inputWidth + x1];
    uchar3 p4 = d_inputImage[y2 * inputWidth + x2];

    uchar3 interpolated;
    // 插值计算
    interpolated.x = (uchar)((1 - wx) * (1 - wy) * p1.x + wx * (1 - wy) * p2.x + (1 - wx) * wy * p3.x + wx * wy * p4.x);
    interpolated.y = (uchar)((1 - wx) * (1 - wy) * p1.y + wx * (1 - wy) * p2.y + (1 - wx) * wy * p3.y + wx * wy * p4.y);
    interpolated.z = (uchar)((1 - wx) * (1 - wy) * p1.z + wx * (1 - wy) * p2.z + (1 - wx) * wy * p3.z + wx * wy * p4.z);
    return interpolated;
}

__global__ void bilinearInterpolationKernel(
    uchar3* d_inputImage,
    uchar3* d_outputImage,
    int inputWidth, int inputHeight,
    int outputWidth, int outputHeight,
    float scaleX, float scaleY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < outputWidth && y < outputHeight) {
        // 计算在源图像中位置
        float srcX = x * scaleX;
        float srcY = y * scaleY;

        uchar3 interpolated_tmp;
        uchar3 interpolated_tmp2 = bilinearInterpolation_test(srcX, srcY,
            d_inputImage, inputWidth, inputHeight);
        d_outputImage[(y * outputWidth + x)] = interpolated_tmp2;
    }
}