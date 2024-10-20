from ultralytics import YOLO
import onnx

model = YOLO("best.pt")
model.export(format="onnx",imgsz=640 , simplify=True)



