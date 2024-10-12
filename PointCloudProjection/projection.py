import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

# 相机内参矩阵
camera_matrix = np.array([
    [4350.62553621579, 0, 2012.54585508037],
    [0, 4376.17813277423, 1518.56324575059],
    [0, 0, 1]
])

# 畸变系数
distortion_coefficients = np.array([0, 0, 0, 0, 0])

# 相机外参矩阵
camera_extrinsic_matrix = np.array([
    [-0.0191969, -0.999807, -0.00427261, 0.015478],
    [0.0178871, 0.00392928, -0.999833, -0.0128704],
    [0.999891, 0.0121324, -0.00835955, -0.691251],
    [0, 0, 0, 1]
])

file_path = '点云投影/cloud.csv'


def read_point_cloud_from_csv(file_path):
    # 使用pandas读取CSV文件
    df = pd.read_csv(file_path)
    # 假设CSV文件中的列名为'x', 'y', 'z'，提取对应的列
    points_3d = df[['X', 'Y', 'Z']].to_numpy()
    intensities = df['Reflectivity'].to_numpy()
    return points_3d, intensities


points_3d, intensities = read_point_cloud_from_csv(file_path)


def project_points(points_3d, camera_matrix, camera_extrinsic_matrix):
    # 将3D点投影到相机坐标系
    points_camera = np.dot(camera_extrinsic_matrix, np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T).T[:, :3]
    # 将相机坐标系的点投影到图像坐标系
    points_2d = np.dot(camera_matrix, points_camera.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    return points_2d


def map_intensity_to_color(intensity):
    # 将反射率强度映射到颜色
    intensity_normalized = intensity / np.max(intensities)
    # 选择颜色映射，这里我们使用一个简单的彩虹色映射
    color_map = plt.get_cmap('hsv')(intensity_normalized)
    color = np.array(color_map[:3]) * 255  # 转换为RGB颜色
    color = tuple(int(c) for c in color)
    return color  # 确保颜色值是整数类型


def map_distance_to_size(distance):
    # 将距离映射到点的大小
    # 这里只是一个示例，您可以根据需要定义自己的映射
    # 确保点的大小在合理的范围内
    size = int( distance / 10)  # 假设距离被标准化到0-1    16.618864040601572
    return size  # 确保点的大小至少为1


points_2d = project_points(points_3d, camera_matrix, camera_extrinsic_matrix)
#
#
#
image = np.zeros((3036, 4024, 3), dtype=np.uint8)
for point, indensity in tqdm(zip(points_2d, intensities)):
    x, y = int(point[0]), int(point[1])
    if 0 <= x < 4024 and 0 <= y < 3036:
        color = map_intensity_to_color(indensity)
        size = map_distance_to_size(point[2]) #Point[2]提供了点的深度信息
        cv2.circle(image, (x,y), size, color, -1)
cv2.imshow('Projected Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()