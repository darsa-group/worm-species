from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data.yaml", epochs=1000, imgsz=1024, batch=6)