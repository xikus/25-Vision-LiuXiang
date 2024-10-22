- 阅读PointNet论文：
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation，只读1-4部分；
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space，只读1-3部分；
- 形成一篇笔记，大致介绍下：
  1. PointNet是如何解决点云的无序性问题的？
  2. 什么是特征对齐模块？
  3. PointNet++在上一代的基础上做了什么改进？


### 点云的无序性
![alt text](image.png)
![alt text](image-2.png)
![alt text](image-1.png)
`we approximate h by a multi-layer perceptron network and g by a composition of a single variable function and a max pooling function.`
- 对每一个`point`,都经过`MLP`进行处理,也就是`(1)`中的`h`,从而得到```feature```
- 在```feature```的各个维度上执行```max pooling```操作
- 作者除`max pooling`外还测试了`mean pooling`和 `weighted sum pooling`, `max pooling`效果最好
### 特征对齐模块
  `The semantic labeling of a point cloud has to be invariant if the point cloud undergoes certain geometric transformations, such as rigid transformation. We therefore expect that the learnt representation by our point set is invariant to these transformations.`
  
-   