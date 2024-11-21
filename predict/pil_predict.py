#!/usr/bin/env python3
"""
@brief pil_predict.py run a simple Predict Mode script with a pretrained YOLO11 model
@version 0.1
@date 2024-10-20
@author armw

@brief Run inference on an image opened with Python Imaging Library (PIL)

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
pil_predict.py

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
from PIL import Image
from ultralytics import YOLO

model_name = "yolo11n"      # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}.pt")    # the nano model by Ultralytics
image_file = "../images/tongliboats.jpg"    # image file to be read
assert os.path.isfile(image_file), f"{image_file} not found"
image_name = Image.open(image_file) # using PIL function to read image
results = model(image_name, stream=True)    # using a parameter driven value for input source

                            # Process results list
for result in results:
    """
    boxes:
        cls: tensor([14.])
        conf: tensor([0.2882])
        data: tensor([[6.2852e+02, 2.1980e+03, 1.8514e+03, 4.4705e+03, 2.8822e-01, 1.4000e+01]])
        id: None
        is_track: False
        orig_shape: (5376, 3024)
        shape: torch.Size([1, 6])
        xywh: tensor([[1239.9497, 3334.2563, 1222.8691, 2272.5525]])
        xywhn: tensor([[0.4100, 0.6202, 0.4044, 0.4227]])
        xyxy: tensor([[ 628.5152, 2197.9802, 1851.3843, 4470.5327]])
        xyxyn: tensor([[0.2078, 0.4089, 0.6122, 0.8316]])
    """
    boxes = result.boxes    # Boxes object for bounding box outputs

    masks = result.masks    # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs    # Probs object for classification outputs
    obb = result.obb        # Oriented boxes object for OBB outputs

    result.show()           # display to screen
    result_file, _ = os.path.splitext(os.path.basename(result.path))  # obtain the filepath
    result_file = result_file + "_" + model_name + ".jpg"  # synthesize the filename
    result.save(filename=result_file)  # save to disk
    for box in result.boxes:    # iterate through all boxes objects in the results object
        for label in box.cls:   # demonstration to check if a specific label was detected
            print(f"{result.names[int(label)]} {box.conf[0]:.2f}")
            if result.names[int(label)] == "person":  # interrogate persons
                print("Halt! Wer da?")    # example to substitute print statement with an alert function