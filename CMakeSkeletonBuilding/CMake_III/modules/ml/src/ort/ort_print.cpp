/**
 * @file ort_print.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-02-04
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <iostream>

#include "srvl/ml/ort.h"

using namespace std;
using namespace Ort;

void OnnxRT::printModelInfo()
{
    cout << "-------------- Input Layer --------------" << endl;
    int input_node = __pSession->GetInputCount();
    cout << "the number of input node is: " << input_node << endl;
    for (int i = 0; i < input_node; i++)
    {
        cout << "[" << i << "]\t┬ name is: " << __pSession->GetInputName(i, __allocator) << endl;
        vector<int64_t> input_dims = __pSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "\t│ dim is: [";
        for (auto dim : input_dims)
        {
            cout << dim << ", ";
        }
        cout << "\b\b]\n";
        cout << "\t└ type of each element is: "
             << __pSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() << endl;
    }

    cout << "\n------------- Output  Layer -------------" << endl;
    int output_node = __pSession->GetOutputCount();
    cout << "the number of output node is: " << __pSession->GetOutputCount() << endl;
    for (int i = 0; i < output_node; i++)
    {
        cout << "[" << i << "]\t┬ name is: " << __pSession->GetOutputName(i, __allocator) << endl;
        vector<int64_t> output_dims = __pSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "\t│ dim is: [";
        for (auto dim : output_dims)
        {
            cout << dim << ", ";
        }
        cout << "\b\b]\n";
        cout << "\t└ type of each element is: "
             << __pSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType() << endl;
    }
}
