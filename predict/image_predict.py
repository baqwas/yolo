#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@package yolo11_tests

@version 0.1
@date 2024-10-20
@author armw


@brief This script runs YOLO11 operations to detect and predict a list of images in specified folder.

Copyright (c) 2023 ParkCircus Productions; All Rights Reserved.

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
image_predict <image filename> -m <YOLO11 model suffix character>

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
"""
import argparse
import os
from ultralytics import YOLO

def image_predict(args):
    """
    Performs basic YOLO detection and predict on specified images with a selected model

    :param args: command line parameters
    :return: True if processing completed successfully
    """

    folder = args.input_folder                  # command line parameter processing

    model_name = "yolo11" + args.model + ".pt"  # if -m | --model is unused on command line then args.model="n"
    model = YOLO(model_name)                    # using a Ultralytics pre-trained model for YOLO11

    results = model([folder])                   # perform detection and prediction

    for result in results:                      # Process results list
        boxes = result.boxes                    # Boxes object for bounding box outputs
        masks = result.masks                    # Masks object for segmentation masks outputs
        keypoints = result.keypoints            # Keypoints object for pose outputs
        probs = result.probs                    # Probs object for classification outputs
        obb = result.obb                        # Oriented boxes object for OBB outputs
        result.show()                           # display to screen
        name_only = model_name.find(".")        # remove the ".pt" suffix from model name
        savefile = (os.path.splitext(os.path.basename(folder))[0] + "_" +
                    model_name[:name_only] + ".jpg")
        result.save(filename=savefile)          # save to disk

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="detect and predict images in specified folder using a YOLO11 model",
        epilog="YOLO11 in action!"
    )
    parser.add_argument(                        # only one input image file, please
        "input_folder",
        type=str, default="../images/royalswans1.jpg",
        help="folder for images for detection and prediction"
    )
    parser.add_argument(
        "-m", "--model", default="n", # n for yolo11n.pt model
        type=str,
        help="model name suffix"
    )

    my_args = parser.parse_args()
    image_predict(my_args)