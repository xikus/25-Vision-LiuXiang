![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

## 装甲板关键点检测
***
- **代码思路**
> 本次任务分为两个部分: *模型训练* 和 *模型部署*

> 使用到的模型包括: 基于YOLOv8-pose的关键点检测模型和数字识别模型

> 模型部署指将.engine格式的模型使用C++部署, 以实现更快的推理速度

### 模型训练
***
- **关键点识别模型**
  - **使用的模块**
    >  ````python
    >  model.py # 训练模型
    >  ultralytics # yolo模型仓库```

    - **模型架构**
    > 模型基于YOLOv8-pose,主要修改为:
    >> 1. 将17个keypoint改为4个, 依次为装甲板左上角, 右上角, 右下角, 左下角.
    >> 2. 修改keypoint的坐标为[x, y], 删去visible, 在`data.yaml`里设置`kpt_shape: [4, 2]`
    >> 3. 检测的class为**红**和**蓝**
    - **数据集**
    > 一份图片放于`images`文件夹下, 一份标签放于`labels`文件夹下, 两者文件名相同

    > 标注格式:
    >> 1. 第0位为`class`(0:红, 1:蓝)
    >> 2. 第1-4位标定装甲板的`bounding box`, 四个数的顺序是x, y, w, h. (x, y)为 `bounding box` 中心点的归一化坐标, w, h为`bounding box`的归一化相对宽高
    >> 3. 第5-13位为`keypoint`的坐标, 顺序如下
    >>> 1. 左上角归一化坐标`x1`,`y1`
    >>> 2. 右上角归一化坐标`x2`,`y2`
    >>> 3. 右下角归一化坐标`x3`,`y3`
    >>> 4. 左下角归一化坐标`x4`,`y4`
  
    - **遇到的问题及解决**
    
    > `warning:WARNING ⚠️ no model scale passed. Assuming scale='n'`
      >> 把`yolov8-pose.yaml`改为`yolov8n-pose.yaml`. 要用到模型的yaml文件的时候, 要么在文件里面scales里面写上n,s,m,l...，要么命名里面加上n,s,m,l...，不写的话默认用n模型.
      >>```python
      >>    scales: # model compound scaling constants, i.e. 'model=yolov8n-pose.yaml' will call yolov8-pose.yaml with scale 'n'
      >>    # [depth, width, max_channels]
      >>    n: [0.33, 0.25, 1024]
      >>    s: [0.33, 0.50, 1024]
      >>    m: [0.67, 0.75, 768]
      >>    l: [1.00, 1.00, 512]
      >>    x: [1.00, 1.25, 512]
      >>     #cited from yolov8-pose.yaml
      >>```
    
    > 在`~/.config/Ultralytics`里需要自行修改数据集路径
  
    - **训练方法**
    > `python3 model.py`

***

- **数字识别模型**
  - **使用的模块**
    >```bash
    >recognize_colored
    >├── model_num.py
    >├── predict_num.py
    >├── __pycache__
    >└── train.py
    >```
  
  - **模型架构**
  > 模型使用两个`block`和一个`fc层`
  >> 1. 每个`block`由一个`conv2d`, 一个`maxpooling2d`, 一个`ReLU`以及一个`BatchNorm`组成
  >> 2. 输出层维度为6
  >> 3. 对每份图像都进行了`transforms.Normalize((0.3000, 0.3020, 0.4224), (0.2261, 0.2384, 0.2214))`处理

  - **数据集**
  > 数据集来自师兄提供, 来源为从4点模型数据集的图片中裁取的装甲板图像

  - **训练方法**
  > 1. `cd recognized_colored`
  > 2. `python3 train.py`


### 模型部署

***

- **导出装甲板识别模型**
> 1. `python3 pt_to_onnx` 导出onnx格式的模型
> 2. `cd /home/yoda/TensorRT-8.6.1.6/bin`
> 3. `trtexec --onnx=best.onnx --explicitBatch --saveEngine=best.engine --workspace=1024 --best`

- **导出数字识别模型**
> 1. ```python
>   torch_input = torch.randn(1, 3, 80, 80)
>   onnx_program = torch.onnx.export(model_num, torch_input, "model_num.onnx", export_params=True)
>    # predict_num.py中的这部分实现导出.onnx格式的数字识别模型
>   ```
> 2. `trtexec --onnx=model_num.onnx --explicitBatch --saveEngine=model_num.engine --workspace=1024 --best`
***
- **用C++实现模型部署**
 >此部分内容全部位于`trt_deploy`

 >```bash
> # trt_deploy文件结构图
> trt_deploy
> ├── best.engine
> ├── cmake
> │   ├── FindTensorRT.cmake
> │   └── Function.cmake
> ├── CMakeLists.txt
> ├── include
> │   ├── common.hpp
> │   ├── filesystem.hpp
> │   ├── num_recog.cpp
> │   └── yolov8-pose.hpp
> ├── main.cpp
> └── model_num.engine
> ```
> best.engine & model_num.engine : 导出的两个模型
> 
> cmake : 用来寻找TensorRT
> 
> include:
>> 1. common.cpp : Logger, Object, get_size_by_dims()等一些底层数据结构和API
 > >
>> 2. filesystem.cpp : A C++17-like filesystem implementation for C++11/C++14/C++17/C++20
 > >
> > 3. num_recog.cpp : 数字识别的相关实现
> > 4. yolov8-pose.hpp : 装甲板识别的相关实现
> >
> main.cpp : 主程序, 实现对照片进行推理的功能

***

- **装甲板识别详解**
> 装甲板识别包括预处理, 模型加载, 推理, 后处理四部分
> > 预处理:
> >> 1. 将图片通过填充的方式变为合适的大小并转换为blob格式(`letterbox`)
> >> 2. 将图片从cpu上copy到gpu上(`copy_from_Mat`)
> 
> > 模型加载:
> >> 1. 创建`Logger`, `Runtime`, `Engine`, `Context`
> >> 2. 通过`engine`获得`BindingDataType`, `BindingName`, 确定`Input`和`Output`, 并将`Input`和`Output`分别`push_back`入`input_bindings`和`output_bindings`.
> >> 3. 通过`engine->getProfileDimensions`得到`Input`的`Dims`, 并通过`context->setBindingDimensions`为`Context`设定其`Dims`.
> >> 4. 为每一个`Output`在GPU(`cudaMallocAsync`)和CPU(`cudaHostAlloc`)上分别划分一片内存空间. 为每一个`Input`在CPU(`cudaMallocAsync`)上分配内存空间.
>
>> 推理:
>>> 1. 通过`context->enqueueV2`进行推理
>>> 2. 通过`cudaMemcpy`将GPU上的`Output`copy到CPU上来
>>> 3. `cudaStreamSynchronize`(前述的推理过程在CPU和GPU之间异步执行)
>
>> 后处理:
>>> 1.从推理产生的8400个框中通过`conf`, `IoU(NMS)`等`metric`筛选出合适的`Bounding Box`.
>>> 2. 将筛选得到的`Bounding Box`的关键点及其信息画到图像上


- **数字识别详解**
> 1. 从图像中裁出装甲板识别模型识别出的装甲板图像
> 2. 投入数字识别模型
> 3. 数字识别模型的输出是一个1*6的数组,每个元素依次表示每个标签的置信度
> 4. 后处理的流程为找出置信度最高的元素的索引

***
- **操作指南**
> ```bash
> cd trt_deploy
> mkdir build
> cd build
> cmake ..
> make
> ./yolov8-pose ../best.engine /home/yoda/25-Vision-LiuXiang/KeyPointsDetection/4PointsModel/images/train/0231.jpg ../model_num.engine
># 第一个参数为装甲板识别模型, 第二个参数是输入的图片, 第三个参数是数字识别的模型
> ```

***
- **经验之谈**
> 使用四点模型时一定要注意`bounding box`的位置,因为nms的iou是根据bounding box来算的,如果`bounding box`超出了图像, 很有可能使得其长宽为负数, 从而导致`IoU`计算出错, 影响`NMS`.


### 总结
- **魔改一些开源模型时可以先粗略了解其结构, 然后上手摸索. 等遇到瓶颈时在仔细学习文档或相关理论. 这样既可以保证足够的效率和实践经历, 也可以保证"知其所以然", 深入领会其理论与Idea.**

- **模型部署可以大幅提高模型推理速度. 但由于涉及到CPU和GPU的内存, 实际操作较难入门. 需要理解CPU和GPU之间的通信和异步的数据传输. 还有CUDA的相关概念, 如Logger, Engine, Context等, 既需要参考官方文档学习, 也需要在实践中慢慢摸索.**