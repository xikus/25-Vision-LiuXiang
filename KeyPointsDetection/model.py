from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8-pose.yaml")  # build a new model from YAML
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="4PointsModel/data.yaml", epochs=100, imgsz=640, device=0, batch=-1)