#!/usr/bin/env python
"""
notes:
The following are some notable features of YOLO26's Train mode:
* **Automatic Dataset Download**: Dataset configurations with a download source are downloaded automatically on first use, e.g., yolo train data=coco8.yaml. See the Datasets overview for supported formats and datasets.
* **Multi-GPU Support**: Scale your training efforts seamlessly across multiple GPUs to expedite the process.
* **Hyperparameter Configuration**: The option to modify hyperparameters through YAML configuration files or CLI arguments.
* **Visualization and Monitoring**: Real-time tracking of training metrics and visualization of the learning process for better insights.

Results saved to /home/reza/PycharmProjects/yolo/runs/detect/train-2

references:
    * https://docs.ultralytics.com/modes/train
"""
from ultralytics import YOLO

# Load a model
# model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)