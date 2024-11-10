#!/usr/bin/env python3
"""
@brief video_predict.py run a simple Predict Mode script with a pretrained YOLO11 model
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
video_predict.py

result.names: {0: 'person', ... 79: 'toothbrush'}
result.boxes.cls: [label id1, ... label idn]

tested images:
https://ultralytics.com/images/bus.jpg


@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

Adapted from: https://docs.ultralytics.com/tasks/detect/#predict
"""
import os
import cv2
from ultralytics import YOLO

model_name = "yolo11n"                              # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}.pt")                    # the nano model by Ultralytics
source_video = "/home/reza/PycharmProjects/yolo11/videos/elephant_train.mov"   # input source for inference
if not os.path.isfile(source_video):                # does the source video exist?
    print(f"Unable to read image file {source_video}")
    exit(-1)

myCapture = cv2.VideoCapture(source_video)          # leverage OpenCV to access the source video
while myCapture.isOpened():                         # process if frames exist
    success, frame_current = myCapture.read()       # frame-by-frame processing
    if success:                                     # frame was read
        results = model(frame_current)              # run inference on current frame
        frame_annotated = results[0].plot()         # prepare frame for display
        cv2.imshow(source_video, frame_annotated)   # On Screen!

        if cv2.waitKey(1) & 0xFF == ord('q'):       # wait for a ms for the letter 'q'
            break
    else:
        break

myCapture.release()                                 # release video capture resources
cv2.destroyAllWindows()                             # Good Housekeeping
print("All done!")
