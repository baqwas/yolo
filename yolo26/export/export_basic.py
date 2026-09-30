from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model
model = YOLO("/home/reza/PycharmProjects/yolo/runs/detect/train-2/weights/best.pt")  # load a custom-trained model

# Export the model
model.export(format="onnx")