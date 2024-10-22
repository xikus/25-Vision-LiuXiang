#include <fstream> 
#include <iostream> 
#include "common.hpp"
#include <NvInfer.h>
#include "yolov8-pose.hpp"

#define CHECK(status) \
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

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 80;
static const int IN_W = 80;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

void doInference(IExecutionContext & context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine. 
    // Engine requires exactly IEngine::getNbBindings() number of buffers. 
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors. 
    // Note that indices are guaranteed to be less than IEngine::getNbBindings() 
    const int inputIndex = engine.getBindingIndex(IN_NAME);
    std::cout<<"inputIndex:"<<inputIndex<<std::endl;
    const int outputIndex = engine.getBindingIndex(OUT_NAME);
    std::cout<<"outputIndex:"<<outputIndex<<std::endl;

    // Create GPU buffers on device 
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 1 * 6 * sizeof(float)));

    // Create stream 
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // input batch data to device, infer on the batch asynchronously, and output back to host 
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 1 * 6 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers 
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

void read_model(const char* engine_name, char* trtModelStream, size_t& size) {
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
}

void recognize(float* input, float* output)
{
    char* trtModelStream;
    size_t size;
    read_model("../model_num.engine", trtModelStream, size);

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
        cv::dnn::blobFromImage(dst, dst, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
        res.push_back(dst);
    }
}