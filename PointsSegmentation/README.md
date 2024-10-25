![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

### 点云分割

- **文件结构**
>```python
>Pointnet_Pointnet2_pytorch
>├── data
>├── data_utils
>├── LICENSE
>├── log
>├── models
>├── provider.py
>├── __pycache__
>├── README.md
>├── test_classification.py
>├── test_partseg.py
>├── test_semseg.py
>├── train_classification.py
>├── train_partseg.py
>├── train_semseg.py
>└── visualizer
>```
>data: 存放数据集
> 
>train_partseg.py: 训练点云分割模型
>
> test_partseg.py: 测试点云分割模型

- **运行环境**
>```python
>python version = 3.9
>pytorch = 2.5.0
>cuda = 11.8
>
- **训练代码**
> ```bash
> python3 train_partseg.py --model pointnet2_part_seg_ssg
>```
#### 总结
- 本项目复现了PointNet++的点云分割模型，实现了对点云数据的分割任务.
- 作者提供的代码和readme存在诸多问题,如:
>1. 数据集路径错误,导致识别到的数据集为空
>2. 数据集标签是未带normal的, 但是示例训练bash脚本启用了normal,导致报错
导致在复现过程中遇到了很多问题, 但也锻炼了python调试的能力.
- 其他细节请参阅论文原文和作者的README.md