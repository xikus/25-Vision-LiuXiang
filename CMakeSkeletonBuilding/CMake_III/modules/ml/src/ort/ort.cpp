/**
 * @file ort.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief the deployment library of the ONNXruntime (Ort)
 * @version 1.0
 * @date 2022-02-04
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <numeric>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#ifdef HAVE_OPENCV_DNN
#include <opencv2/dnn.hpp>
#endif //! HAVE_OPENCV_DNN

#include "srvl/ml/ort.h"
#include "srvl/core/util.hpp"

#include "srvlpara/ml/ort.h"

using namespace cv;
using namespace std;
using namespace Ort;
using namespace para;

OnnxRT::OnnxRT(const string &model_path) : __env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "OnnxDeployment"),
                                           __memory_info(MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                               OrtMemType::OrtMemTypeDefault))
{
    if (model_path.empty())
        SRVL_Error(SRVL_StsBadArg, "Model path is empty!");
    setupEngine(model_path);
}

void OnnxRT::setupEngine(const string &model_path)
{
#ifdef WITH_ORT_CUDA
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif // WITH_ORT_CUDA

#ifdef WITH_ORT_TensorRT
    OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);
#endif // WITH_ORT_TensorRT

    __session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    __pSession = make_unique<Session>(__env, model_path.c_str(), __session_options);

    // define the names of the I/O nodes
    for (size_t i = 0; i < __pSession->GetInputCount(); i++)
        __input_names.emplace_back(__pSession->GetInputName(i, __allocator));
    for (size_t i = 0; i < __pSession->GetOutputCount(); i++)
        __output_names.emplace_back(__pSession->GetOutputName(i, __allocator));
    // setup input array
    __input_arrays.resize(__pSession->GetInputCount());
    for (size_t i = 0; i < __pSession->GetInputCount(); i++)
    {
        auto shape = __pSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        __input_arrays[i].resize(shape[1] * shape[2] * shape[3]);
    }
}

vector<size_t> OnnxRT::inference(const vector<Mat> &images)
{
    vector<Value> input_tensors = preProcess(images);
    vector<Value> output_tensors = doInference(input_tensors);
    return postProcess(output_tensors);
}

vector<Value> OnnxRT::preProcess(const vector<Mat> &images)
{
    size_t input_count = __pSession->GetInputCount();
    if (input_count != images.size())
        CV_Error(SRVL_StsBadArg, "Size of the \"images\" are not equal to the model input_count.");
    // get the correct data of each input layer
    vector<Value> input_tensors;
    for (size_t i = 0; i < input_count; i++)
    {
        auto input_shape = __pSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        // [2], [3] are the correct size of the image
        if (input_shape.size() != 4)
            SRVL_Error(SRVL_StsBadSize, "Size of the input_shape of the model is not equal to \'4\'");
        if (input_shape[2] != input_shape[3])
            SRVL_Error(SRVL_StsError, "Shape of the input_shape[2] and input_shape[3] of the model is unequal");
        input_shape[0] = 1;
        // update the size of each input layer
        Mat input_image;
        resize(images[i], input_image, Size(input_shape[2], input_shape[3]));
        // allocate memory and normalization
        imageToVector(input_image, __input_arrays[i]);
        input_tensors.emplace_back(Value::CreateTensor<float>(__memory_info,
                                                              __input_arrays[i].data(), __input_arrays[i].size(),
                                                              input_shape.data(), input_shape.size()));
    }
    return input_tensors;
}

vector<size_t> OnnxRT::postProcess(vector<Value> &output_tensors)
{
    // 所有输出对应的置信度最高的索引
    vector<size_t> output_indexs;
    for (auto &output_tensor : output_tensors)
    {
        // 获取置信度最高的索引
        const float *output = output_tensor.GetTensorData<float>();
        vector<size_t> indexs(output_tensor.GetTensorTypeAndShapeInfo().GetElementCount());
        iota(indexs.begin(), indexs.end(), 0);
        auto it = max_element(indexs.begin(), indexs.end(),
                              [&output](size_t lhs, size_t rhs)
                              {
                                  return output[lhs] < output[rhs];
                              });
        output_indexs.emplace_back(*it);
    }
    return output_indexs;
}

void OnnxRT::imageToVector(Mat &input_image, vector<float> &input_array)
{
    // CHW
    int C = input_image.channels();
    if (C != 1 && C != 3)
        SRVL_Error_(SRVL_StsBadArg, "Bad channel of the input argument: \"input_image\", chn = %d", C);
    int H = input_image.rows;
    int W = input_image.cols;
    size_t pixels = C * H * W;
    if (pixels != input_array.size())
        SRVL_Error(SRVL_StsBadArg, "The size of the arguments: \"input_image\" and \"input_array\" are not equal");
#ifdef HAVE_OPENCV_DNN
    Mat blob_image;
    // 转 Tensor 的 NCHW 格式，做归一化和标准化
    if (C == 1)
    {
        double scalar_factor = 1.0 / (255.0 * ort_param.MONO_STDS);
        double means = 255.0 * ort_param.MONO_MEANS;
        blob_image = dnn::blobFromImage(input_image, scalar_factor, Size(), means, true);
    }
    else
    {
        vector<Mat> channels; // input_image 所有通道
        split(input_image, channels);
        vector<Mat> blob_channels(channels.size()); // blob 所有通道
        Vec3d scalar_factor = {1.0 / (255.0 * ort_param.RGB_STDS(0)),
                               1.0 / (255.0 * ort_param.RGB_STDS(1)),
                               1.0 / (255.0 * ort_param.RGB_STDS(2))};
        Vec3d means = 255.0 * ort_param.RGB_MEANS;
        for (size_t i = 0; i < channels.size(); ++i)
            blob_channels[i] = dnn::blobFromImage(channels[i], scalar_factor(i), Size(), means(i), true);
        merge(blob_channels, blob_image);
    }
    memcpy(input_array.data(), blob_image.ptr<float>(), sizeof(float) * static_cast<float>(pixels));
#else
    vector<double> means(C);
    vector<double> stds(C);
    if (C == 1)
    {
        means[0] = ort_param.MONO_MEANS;
        stds[0] = ort_param.MONO_STDS;
    }
    else
        for (int i = 0; i < C; ++i)
        {
            means[i] = ort_param.RGB_MEANS[i];
            stds[i] = ort_param.RGB_STDS[i];
        }
    // 转 Tensor 的 NCHW 格式，做归一化和标准化
    float *p_input_array = input_array.data();
    for (int c = 0; c < C; c++)
    {
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                p_input_array[c * H * W + h * W + w] = input_image.ptr<uchar>(h)[w * C + 2 - c] - means[c] / 255.f;
                p_input_array[c * H * W + h * W + w] = (p_input_array[c * H * W + h * W + w] - means[c]) / stds[c];
            }
        }
    }
#endif //! HAVE_OPENCV_DNN
}
