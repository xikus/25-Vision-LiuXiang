from ultralytics import YOLO
import onnx

model = YOLO("best.pt")
model.export(format="engine", device=0)



