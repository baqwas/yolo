#!/usr/bin/env python3
"""
@brief plot_predict.py use the plot method to simpify visualization of results object
@version 0.1
@date 2024-10-20
@author armw

@brief This script will infer objects in the input source using YOLO11

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
plot_predict.py

plot() method parameters are documented in the following link:
@sa https://docs.ultralytics.com/modes/predict/#plotting-results

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

Adapted from: https://docs.ultralytics.com/tasks/detect/#predict
"""
import os
from PIL import Image
from ultralytics import YOLO

model_name = "yolo11n"      # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}.pt")    # the nano model by Ultralytics
images = ["/home/reza/PycharmProjects/yolo11/images/macaws.jpg",
    "/home/reza/PycharmProjects/yolo11/images/birds.jpg"] # input source for inference
for image_name in images:
    if not os.path.isfile(image_name):
        print(f"Unable to read image file {image_name}")
        exit(-1)
results = model(images)    # using a parameter driven value for input source

for index, result in enumerate(results):   # process the results list
    image_bgr = result.plot()   # note use of of BGR order numpy array
    image_rgb = Image.fromarray(image_bgr[..., ::-1])   # PIL prefers RGB order array
    result.show()               # show the result
    result.save(filename=f"results{index}.jpg")