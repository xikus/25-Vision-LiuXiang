#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>
#include <cuda_runtime.h>
typedef unsigned char uchar;

unsigned char* read_image(const char* filename);
void bilinearInterpolation_launch(uchar3* h_inputImageUChar3,
    uchar3* h_outputImageUChar3,
    int inputWidth, int inputHeight,
    int outputWidth, int outputHeight);
