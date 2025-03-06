#!/usr/bin/env python3
"""
@brief cls_predict.py run a simple Predict Mode script with a pretrained YOLO11 model
@version 0.1
@date 2024-10-20
@author armw

@brief This script will classify objects in the input source using YOLO11

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
cls_predict.py

Probs:
Attributes:
    data, orig_shape, top1, top5, top1conf, top5conf
Methods:
    cpu, numpy, cuda, to
Adapted from:
https://docs.ultralytics.com/tasks/segment/#val

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

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
"""
import os
from ultralytics import YOLO

model_name = "yolo11n-cls"      # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}.pt")    # the nano model by Ultralytics
image_name = "../images/royalswans/swan04.jpg"   # input source for inference
if not os.path.isfile(image_name):
    print(f"Unable to read image file {image_name}")
    exit(-1)
results = model(f"{image_name}")    # using a parameter driven value for input source

                            # Process results list
for result in results:
    boxes = result.boxes    # Boxes object for bounding box outputs

    masks = result.masks    # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs    # Probs object for classification outputs
    obb = result.obb        # Oriented boxes object for OBB outputs

    result.show()           # display to screen
    result_file, _ = os.path.splitext(os.path.basename(result.path))  # obtain the filepath
    result_file = result_file + "_" + model_name + ".jpg"  # synthesize the filename
    result.save(filename=result_file)  # save to disk

    for index, top in enumerate(probs.top5):
        print(f"{result.names[top]} {probs.top5conf[index]:.2f}")



