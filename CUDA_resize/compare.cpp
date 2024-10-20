#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

const char* image_path = "./test.jpg";

int main() {
    cv::Mat image = cv::imread(image_path);
    cv::Mat image_resized;
    auto start = std::chrono::high_resolution_clock::now();
    cv::resize(image, image_resized, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenCV resize Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    cv::imwrite("./output.jpg", image_resized);
}