![img](https://i0.hdslb.com/bfs/new_dyn/791944995fff725f42c7f5a9b64f8567100423098.png@1295w.webp)

### 点云投影
- **代码思路**
```python
read_point_cloud_from_csv() #从csv文件中读取点云数据
project_points() #将点云投影到平面上
map_distance_to_color() #将点云的距离映射到颜色上
map_intensity_to_color() #将点云的强度映射到颜色上    
  ```
> 1. 对于每个点,通过相机外参将其从世界坐标系转换到相机坐标系,再通过相机内参将其转换到像素坐标系
> 2. 通过map_distance_to_color()函数将点云的距离映射到颜色上,通过map_intensity_to_color()函数将点云的强度映射到颜色上
> 3. 通过map_distance_to_color()函数将点云的距离映射到颜色上,通过map_intensity_to_color()函数将点云的强度映射到颜色上
> 4. 点与相机的距离可以通过计算出的点齐次坐标的z值得到

- **遇到问题**
 > 速度过慢, 每次运行需要1min左右

- **解决方法**
 > 问题在于点云投影是串行的, 可以通过并行计算加速

- **效果图**
![20241024-202731.jpg](20241024-202731.jpg)

- **使用方法**
```bash
python3 projecton.py
```