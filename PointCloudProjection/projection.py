import cv2
import numpy as np
import pandas as pd

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
    return points_3d


points_3d = read_point_cloud_from_csv(file_path)


def project_points(points_3d, camera_matrix, camera_extrinsic_matrix):
    # 将3D点投影到相机坐标系
    points_camera = np.dot(camera_extrinsic_matrix, np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T).T[:, :3]
    # 将相机坐标系的点投影到图像坐标系
    points_2d = np.dot(camera_matrix, points_camera.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    return points_2d


points_2d = project_points(points_3d, camera_matrix, camera_extrinsic_matrix)


# 绘制投影点
image = np.zeros((3036, 4024, 3), dtype=np.uint8)
for point in points_2d:
    x, y = int(point[0]), int(point[1])
    if 0 <= x < 4024 and 0 <= y < 3036:
        image[y, x] = (255, 0, 0)

cv2.imshow('Projected Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()