# predict class and key points, and offer the key points to draw_boxes and predict_number
import numpy as np
import predict_num
from ultralytics import YOLO
import cv2



def wrap_perspective(image, points):
    # 定义四个点的新位置，这里我们将其映射到一个80x80的正方形
    # 注意：这四个点的顺序应该与输入的四个点的顺序相同
    dst_points = np.array([[0, 0], [0, 80], [80, 80], [80, 0]], dtype=np.float32)

    # 获取输入点的坐标
    src_points = np.array(points, dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    warped_image = cv2.warpPerspective(image, M, (80, 80))

    cv2.imshow("warped_image", warped_image)
    return warped_image


#

def predict_number(image, points):
    image = wrap_perspective(image, points)
    pred_number = predict_num.predict_number(image)
    return pred_number


def design_boxes(image, keypoints_sets, labels_set):
    for points, label in zip(keypoints_sets, labels_set):
        predicted_number = predict_number(image, points)
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # 标注class和预测数字
        color = 'blue' if label == 1.0 else 'red'
        cv2.putText(image, "{}".format(color), points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, "{}".format(predicted_number), points[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image


# Load a model
model = YOLO("best.pt")  # pretrained YOLO11n model
predict_photos = ['4PointsModel/images/val/326.jpg']
# Run batched inference on a list of images
results = model(predict_photos)  # return a list of Results objects

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     orig_img = result.orig_img  # original image
#     keypoints = result.keypoints.data.cpu().numpy()  # keypoints object
#     number = predict_number(orig_img, keypoints)
#     annotated_point = (int(keypoints[0][3][0]), int(keypoints[0][3][1]))
#     show_img = result.plot(conf=False, line_width=1, font_size=1.5, kpt_line=True, show=False, labels=True)
#     annotator = Annotator(show_img, line_width=2, font_size=20, font='Arial.ttf')
#     annotator.text(annotated_point, "{}".format(number), txt_color=(0, 255, 0))
#     cv2.imshow("result", show_img)
# cv2.waitKey(0)


# Process results list
for result in results:
    orig_img = result.orig_img  # original image
    labels_set = result.boxes.cls.cpu().numpy()  # class labels
    keypoints_sets = result.keypoints.data.cpu().numpy().astype(int)  # keypoints object
    show_img = design_boxes(orig_img, keypoints_sets, labels_set)
    cv2.imshow("result", show_img)
cv2.waitKey(0)
