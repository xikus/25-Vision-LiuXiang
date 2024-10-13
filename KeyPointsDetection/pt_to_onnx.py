from ultralytics import YOLO
import onnx

model = YOLO("best.pt")
model.export(format="onnx", device=0)



