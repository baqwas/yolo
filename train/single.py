#!/usr/bin/env python3
"""
@brief single.py train using an existing dataset
@version 0.1
@date 2024-10-20
@author armw

@brief This script will train a YOLO11 model with the COCO dataset

Copyright (C) 2024 ParkCircus Productions; All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Usage:
simple.py

training data
deep learning models learn by iteratively adjusting their internal parameters (weights & biases)
   to minimize the difference between their predictions and the actual values in the training data

parameter updates
After processing each data point (or a small batch of data points), the model's parameters are updated based on the calculated error

epoch
represents a single cycle where the model processes every single data point in the training set once
are fundamental to the training process, allowing the model to gradually improve its performance
    by iteratively adjusting its parameters

multiple epochs
are required for the model to learn effectively. The number of epochs is a crucial hyperparameter
    that determines the duration of the training process

over or under training
The number of epochs is a hyperparameter that needs to be carefully chosen.
    Too few epochs may lead to underfitting (the model fails to learn the underlying patterns in the data),
    while too many epochs can lead to overfitting (the model performs well on the training data but poorly on new, unseen data).

@sa https://docs.ultralytics.com/modes/train/#introduction

Adapted from: https://docs.ultralytics.com/tasks/detect/#predict
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}

model = YOLO("yolo11n.pt")              # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

"""
from ultralytics import YOLO

                                        # Load a model
model = YOLO("yolo11n.yaml")            # build a new model from YAML


                                        # Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)