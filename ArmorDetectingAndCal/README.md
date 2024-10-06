# ARMORDETECTINGAndCAL
## Intro
**calculate:** 距离和姿态解算
**detect_light:** 检测灯条
**LightDescriptor:** 表示灯条的数据类型
**match_light:** 匹配灯条以寻找装甲板
**test_light_descriptor.cpp:** 测试程序 

## 关键参数
**ENEMY_COLOR:** 敌方灯条颜色. 修改`detect/detect_light/src/detect_light.cpp`的`_enemy_color`参数. BLUE表示敌方灯条为蓝色,RED表示地方灯条为红色.
**VideoPath:** 测试视频路径. 修改`detect/test_light_descriptor.cpp`中的VideoPath.


## 运行

#### 编译步骤
```bash
cd detect
mkdir build
cd build
cmake ..
make -j6
```

#### 运行步骤
```bash
./test
```