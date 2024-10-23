//
// Created by ubuntu on 4/7/23.
//
#ifndef POSE_NORMAL_YOLOv8_pose_HPP
#define POSE_NORMAL_YOLOv8_pose_HPP
#include "NvInfer.h"
#include "common.hpp"
#include <fstream>
using namespace nvinfer1;
using namespace pose;

class YOLOv8_pose {
public:
    explicit YOLOv8_pose(const std::string& engine_file_path);

    ~YOLOv8_pose();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    static void draw_objects(const cv::Mat& image,
        cv::Mat& res,
        const std::vector<Object>& objs,
        const std::vector<std::vector<unsigned int>>& SKELETON,
        const std::vector<std::vector<unsigned int>>& KPS_COLORS,
        const std::vector<std::vector<unsigned int>>& LIMB_COLORS);


    // Each Binding object contains information about an input tensor, such as its name, size, and dimensions.
    int                  num_bindings; // input + output num
    int                  num_inputs = 0; // input num
    int                  num_outputs = 0; // input num
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;

    std::vector<void*>   host_ptrs; // A vector that holds pointers to CPU memory for input and output data.

    std::vector<void*>   device_ptrs; // A vector that holds pointers to GPU memory for input and output data. This memory is used during the inference process.

    PreParam pparam; // An object that holds preprocessing parameters, such as scaling factors and padding values, used during image preprocessing.

private:
    nvinfer1::ICudaEngine* engine = nullptr; // The engine is responsible for executing the neural network.(?)

    nvinfer1::IRuntime* runtime = nullptr; // The runtime is used to deserialize the engine from a file.(?)

    nvinfer1::IExecutionContext* context = nullptr; // The context is used to manage the execution of the engine, including setting input dimensions and running inference.(?)

    cudaStream_t                 stream = nullptr; // A CUDA stream used for asynchronous memory operations and kernel launches. It allows for parallel execution of tasks on the GPU.(?)

    Logger                       gLogger{ nvinfer1::ILogger::Severity::kERROR }; // This logger is used to log messages from the TensorRT library.
};

YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path)
{
    // read the engine_file
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good() && "Custom0 error message");
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream && "Custom1 error message");
    file.read(trtModelStream, size);
    file.close();

    //initLibNvInferPlugins(&this->gLogger, ""); // This is a function provided by TensorRT to initialize any custom plugins that you might want to use in your neural network inference.
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr && "Custom2 error message");

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr && "Custom3 error message");
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr && "Custom4 error message");
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbBindings();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
        std::string        name = this->engine->getBindingName(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype); // the size of the data type

        bool IsInput = engine->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs += 1;
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setBindingDimensions(i, dims); // set the dimensions of the input tensor.
        }
        else {
            dims = this->context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_pose::~YOLOv8_pose()
{
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

// allocate memory for input and output tensors on the GPU and CPU.
void YOLOv8_pose::make_pipe(bool warmup)
{

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void* d_ptr, * h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size = bindings.size * bindings.dsize;
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

// resize an input image to fit within a specified size while maintaining the aspect ratio. also add padding to ensure the final image matches the target dimensions exactly.
void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float       height = image.rows;
    float       width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp1, tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp1, cv::Size(padw, padh));
    }
    else {
        tmp1 = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp1, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 0, 0, 0 });


    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
    ;
}

// take an OpenCV cv::Mat image, preprocess it, and then copy it to a GPU device memory asynchronously using CUDA. part of a pipeline for running inference on a neural network using TensorRT
void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto& in_binding = this->input_bindings[0];
    auto     width = in_binding.dims.d[3];
    auto     height = in_binding.dims.d[2];
    cv::Size size{ width, height };
    this->letterbox(image, nchw, size);

    this->context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, height, width} });

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    this->context->setBindingDimensions(0, nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

// esponsible for running inference using a YOLOv8 model on a GPU.It involves enqueuing a task to the GPU, copying the results back to the host, and synchronizing the CUDA stream.
void YOLOv8_pose::infer()
{

    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        // std::cout << "osize:" << osize << std::endl;
        // std::cout << "size:" << this->output_bindings[i].size << std::endl;
        // std::cout << "dsize:" << this->output_bindings[i].dsize << std::endl;
        CHECK(cudaMemcpy(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost));//, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

// 
void YOLOv8_pose::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors = this->output_bindings[0].dims.d[2];
    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;
    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;
    // std::cout<<this->host_ptrs.size()<<std::endl;
    // cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    // output = output.t();
    for (int i = 0; i < num_anchors; i++) {
        // auto row_ptr = output.row(i).ptr<float>();
        // auto bboxes_ptr = row_ptr;
        // auto scores_ptr = row_ptr + 4;
        // auto kps_ptr = row_ptr + 6;
        auto head_ptr = static_cast<float*>(this->host_ptrs[0]);
        auto bboxes_ptr = head_ptr + i;
        auto scores_ptr_red = head_ptr + 4 * num_anchors + i;
        auto scores_ptr_blue = head_ptr + 5 * num_anchors + i;
        auto scores_ptr = ((*scores_ptr_red) > (*scores_ptr_blue)) ? scores_ptr_red : scores_ptr_blue;
        int color = ((*scores_ptr_red) > (*scores_ptr_blue)) ? 0 : 1;
        auto kps_ptr = head_ptr + 6 * num_anchors + i;
        // for (size_t h = 0; h < 14; h++)
        // {
        //     std::cout << *(head_ptr + h * num_anchors + i) << std::endl;
        // }

        float score = *scores_ptr;
        if (score > score_thres) {
            // std::cout << "score:" << score << std::endl;
            float x = *bboxes_ptr - dw;
            float y = *(bboxes_ptr + num_anchors) - 2 * dh;
            float w = *(bboxes_ptr + 2 * num_anchors) - dw;
            float h = *(bboxes_ptr + 3 * num_anchors) - dh;


            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);
            // std::cout << "x0:" << x0 << " y0:" << y0 << " x1:" << x1 << " y1:" << y1 << std::endl;

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0 + 90;
            bbox.width = abs(x1 - x0);
            bbox.height = abs(y1 - y0) / 4;
            std::vector<float> kps;
            for (int k = 0; k < 4; k++) {
                float kps_x = (*(kps_ptr + 2 * k * num_anchors) - dw) * ratio;
                float kps_y = (*(kps_ptr + (2 * k + 1) * num_anchors) - dh) * ratio;
                kps_x = clamp(kps_x, 0.f, width);
                kps_y = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
            }

            bboxes.push_back(bbox);
            labels.push_back(color);
            scores.push_back(score);
            kpss.push_back(kps);
            // std::cout << "bboxes:" << bboxes.size() << std::endl;


            // Object obj;
            // obj.rect = bbox;
            // obj.prob = score;
            // std::cout<<"prob:"<<obj.prob<<std::endl;
            // obj.label = color;
            // obj.kps = kps;
            // objs.push_back(obj);

        }
        //std::cout << "anchor:" << i << std::endl;
        //std::cout << std::endl;
    }
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);

    // #ifdef BATCHED_NMS
//     cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
// #else
//     cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
// #endif
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect = bboxes[i];
        obj.prob = scores[i];
        // std::cout << "prob:" << obj.prob << std::endl;
        obj.label = labels[i];
        obj.kps = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
    // for (size_t i = 0; i < 100; i++)
    // {
    //     Object obj;
    //     obj.rect = bboxes[i];
    //     obj.prob = scores[i];
    //     std::cout<<"prob:"<<obj.prob<<std::endl;
    //     obj.label = labels[i];
    //     obj.kps = kpss[i];
    //     objs.push_back(obj);
    // }

    std::cout << "the number of armors:" << objs.size() << std::endl;
    // for (size_t i = 0; i < objs.size(); i++)
    // {
    //     std::cout << "point: " << objs[i].kps[0] <<" "<< objs[i].kps[1]<< std::endl;
    // }

    // std::cout << "indices size:" << indices.size() << std::endl;
}

void YOLOv8_pose::draw_objects(const cv::Mat& image,
    cv::Mat& res,
    const std::vector<Object>& objs,
    const std::vector<std::vector<unsigned int>>& SKELETON,
    const std::vector<std::vector<unsigned int>>& KPS_COLORS,
    const std::vector<std::vector<unsigned int>>& LIMB_COLORS)
{
    res = image.clone();
    const int num_point = 4;

    for (auto& obj : objs) {
        int color = obj.label;
        cv::Rect rect = obj.rect;
        std::cout << "rect of armor:" << rect << std::endl;
        // cv::rectangle(res, rect, { 0, 255, 255 }, 3);
        // cv::circle(res, { rect.x, rect.y }, 3, { 255, 255, 255 }, -1);
        // std::cout << "color:" << color << std::endl;
        int tep_x = obj.kps[0];
        int tep_y = obj.kps[1];
        std::cout << "upleft point of armor:" << tep_x << " " << tep_y << std::endl;
        char text[256];
        int      baseLine = 0;
        if (color == 0) {
            // cv::rectangle(res, obj.rect, { 0, 0, 0 }, 2);
            sprintf(text, "red %.1f%%", obj.prob * 100);
        }
        else if (color == 1)
        {
            // cv::rectangle(res, obj.rect, { 255, 0, 0 }, 2);
            sprintf(text, "blue %.1f%%", obj.prob * 100);
        }


        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.kps[0];
        int y = (int)obj.kps[1] - 20;

        if (y > res.rows)
            y = res.rows;


        if (color == 0) {
            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), { 0, 0, 255 }, -1);
        }
        else if (color == 1)
        {
            cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), { 255, 0, 0 }, -1);
        }

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

        auto& kps = obj.kps;
        for (int k = 0; k < num_point; k++) {

            int   kps_x = std::round(kps[k * 2]);
            int   kps_y = std::round(kps[k * 2 + 1]);

            cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
            cv::circle(res, { kps_x, kps_y }, 3, kps_color, -1);

            auto& ske = SKELETON[k];
            int   pos1_x = std::round(kps[ske[0] * 2]);
            int   pos1_y = std::round(kps[(ske[0] * 2) + 1]);

            int pos2_x = std::round(kps[ske[1] * 2]);
            int pos2_y = std::round(kps[(ske[1] * 2) + 1]);

            cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
            cv::line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color, 2);

        }
    }

}

#endif  // POSE_NORMAL_YOLOv8_pose_HPP
