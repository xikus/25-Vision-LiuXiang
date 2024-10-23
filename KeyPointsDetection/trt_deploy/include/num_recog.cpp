#include <fstream> 
#include <iostream> 
#include "common.hpp"
#include <NvInfer.h>
#include "yolov8-pose.hpp"

#define CHECK_NUM(status) \
do\
{\
auto ret = (status);\
if (ret != 0)\
{\
std::cerr << "Cuda failure: " << ret << std::endl;\
abort();\
}\
} while (0)

using namespace nvinfer1;

const char* IN_NAME = "input.1";
const char* OUT_NAME = "26";
static const int IN_H = 80;
static const int IN_W = 80;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
static const cv::Scalar mean(0.3000, 0.3020, 0.4224);
static const cv::Scalar stddev(0.2261, 0.2384, 0.2214);

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine. 
    // Engine requires exactly IEngine::getNbBindings() number of buffers. 
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors. 
    // Note that indices are guaranteed to be less than IEngine::getNbBindings() 
    const int inputIndex = engine.getBindingIndex(IN_NAME);
    const int outputIndex = engine.getBindingIndex(OUT_NAME);

    // Create GPU buffers on device 
    CHECK_NUM(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 1 * 6 * sizeof(float)));

    // Create stream 
    cudaStream_t stream;
    CHECK_NUM(cudaStreamCreate(&stream));

    // input batch data to device, infer on the batch asynchronously, and output back to host 
    CHECK_NUM(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK_NUM(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 1 * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers 
    cudaStreamDestroy(stream);
    CHECK_NUM(cudaFree(buffers[inputIndex]));
    CHECK_NUM(cudaFree(buffers[outputIndex]));
}

// void read_model(const char* engine_name, char* trtModelStream, size_t& size) {
//     // create a model using the API directly and serialize it to a stream
//     std::ifstream file(engine_name, std::ios::binary);
//     if (file.good()) {
//         file.seekg(0, file.end);
//         size = file.tellg();
//         file.seekg(0, file.beg);
//         trtModelStream = new char[size];
//         assert(trtModelStream);
//         file.read(trtModelStream, size);
//         file.close();
//     }
// }

void recognize(float* input, float* output, const char* engine_name)
{
    char* trtModelStream;
    size_t size;
    // create a model using the API directly and serialize it to a stream
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    
    Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);

    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    doInference(*context, input, output, BATCH_SIZE);

    // Destroy the engine 
    context->destroy();
    engine->destroy();
    runtime->destroy();
    delete[] trtModelStream;
}

cv::Mat getTransform(Object& obj) {
    cv::Point2f dst[] = { {0, 0}, {0, 80}, {80, 80}, {80, 0} };
    cv::Point2f src[] = { {obj.kps[0], obj.kps[1]}, {obj.kps[2], obj.kps[3]}, {obj.kps[4], obj.kps[5]}, {obj.kps[6], obj.kps[7]} };
    return cv::getPerspectiveTransform(src, dst);
}

void resize_img(cv::Mat& Image, std::vector<cv::Mat>& res, const std::vector<Object>& objs)
{
    for (auto obj : objs) {
        cv::Mat trans = getTransform(obj);
        cv::Mat dst;
        cv::warpPerspective(Image, dst, trans, cv::Size(80, 80));
        cv::imshow("dst", dst);
        cv::dnn::blobFromImage(dst, dst, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
        dst = dst - mean;
        dst = dst / stddev;

        res.push_back(dst);
    }
}

void findMax(float* input, int& indexMax) {
    int length = BATCH_SIZE * 1 * 6;
    float max = 0.0;
    for (size_t i = 0; i < length; i++)
    {
        if (input[i] > max) {
            max = input[i];
            indexMax = i;
        }
    }
}

void draw_num(cv::Mat image, std::vector<Object>& objs, std::vector<int>& indicesMax) {
    for (size_t i = 0; i < objs.size(); i++)
    {
        cv::putText(image, std::to_string(indicesMax[i]), cv::Point(objs[i].kps[4] + 12, objs[i].kps[5] + 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
    }
}