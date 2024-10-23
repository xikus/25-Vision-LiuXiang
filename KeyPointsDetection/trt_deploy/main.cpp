//
// Created by ubuntu on 4/7/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8-pose.hpp"
#include <chrono>
#include "num_recog.cpp"

namespace fs = ghc::filesystem;

//关键点颜色
const std::vector<std::vector<unsigned int>> KPS_COLORS = { {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0},
                                                           {0, 255, 0} };

//关键点连接顺序
const std::vector<std::vector<unsigned int>> SKELETON = { {0, 1},
                                                         {1, 2},
                                                         {2, 3},
                                                         {3, 0} };

//连接线颜色
const std::vector<std::vector<unsigned int>> LIMB_COLORS = { {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255},
                                                            {51, 153, 255} };

int main(int argc, char** argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    //the path of engine file
    const std::string engine_file_path{ argv[1] };

    //the path of image or video
    const fs::path    path{ argv[2] };

    std::vector<std::string> imagePathList;
    bool                     isVideo{ false };

    assert(argc == 4);

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(false);

    // check the format of the input and prepare the imagePathList or set isVideo to true 
    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png") {
            imagePathList.push_back(path);
        }
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov" || suffix == ".mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }

    //the case of image directory
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat  res, image;
    cv::Size size = cv::Size{ 640, 640 };
    int      topk = 100;
    float    score_thres = 0.2f; //0.2
    float    iou_thres = 0.6f; // 0.8

    std::vector<Object> objs;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            objs.clear();
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.; //time of inference
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.; //time of inference

            //Number Recognition
            std::vector<cv::Mat> num_images;
            std::vector<int> indicesMax;
            resize_img(image, num_images, objs);
            auto start1 = std::chrono::system_clock::now();
            for (auto num_image : num_images) {
                float* output = new float[BATCH_SIZE * 1 * 6];
                auto start = std::chrono::system_clock::now();
                recognize(num_image.ptr<float>(), output, argv[3]);
                auto end = std::chrono::system_clock::now();
                auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.; //time of number recognition
                std::cout << "cost for recognizing one number: " << tc << " ms" << std::endl;
                int indexMax;
                findMax(output, indexMax);
                indicesMax.push_back(indexMax);
                delete[] output;
            }
            //num_image,  objs, indicesMax的排序是一致的

            auto end1 = std::chrono::system_clock::now();
            auto tc1 = (double)std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000.; //time of number recognition

            draw_num(res, objs, indicesMax);

            printf("cost for finding armor %2.4lf ms\n", tc);
            printf("cost for recognizing all number %2.4lf ms\n", tc1);


            cv::imshow("result", res);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8_pose;
    return 0;
}

