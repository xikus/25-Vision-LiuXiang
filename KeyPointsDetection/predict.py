# predict class and key points, and offer the key points to draw_boxes and predict_number
import numpy as np
import predict_num
from ultralytics import YOLO
import cv2

from ultralytics.utils.plotting import Annotator


def wrap_perspective(image, points):
    # 定义四个点的新位置，这里我们将其映射到一个80x80的正方形
    # 注意：这四个点的顺序应该与输入的四个点的顺序相同
    dst_points = np.array([[0, 0], [80, 0], [80, 80], [0, 80]], dtype=np.float32)

    # 获取输入点的坐标
    src_points = np.array(points, dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    warped_image = cv2.warpPerspective(image, M, (80, 80))

    return warped_image
#

def predict_number(image, points):
    image = wrap_perspective(image, points)
    pred_number = predict_num.predict_number(image)
    return pred_number


def design_boxes(image, keypoints, class_label, predicted_number):
    points = keypoints.reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 标注class和预测数字
    cv2.putText(image, class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, predicted_number, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

# Load a model
model = YOLO("best.pt")  # pretrained YOLO11n model
predict_photos = ["497.jpg"]
# Run batched inference on a list of images
results = model(predict_photos)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    orig_img = result.orig_img  # original image
    keypoints = result.keypoints.data.cpu().numpy()  # keypoints object
    number = predict_number(orig_img, keypoints)
    annotated_point = (int(keypoints[0][3][0]), int(keypoints[0][3][1]))
    show_img = result.plot(conf=False, line_width=1, font_size=1.5, kpt_line=True, show=False, labels=True)
    annotator = Annotator(show_img, line_width=2, font_size=20, font='Arial.ttf')
    annotator.text(annotated_point, "{}".format(number), txt_color=(0, 255, 0))
    cv2.imshow("result", show_img)
cv2.waitKey(0)


