#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@package yolo11_tests

@version 0.1
@date 2024-10-20
@author armw

@brief This script runs YOLO11 operations to detect and predict a list of images in specified folder.

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
image_predict <image filename> -m <YOLO11 model suffix character>

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
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

    def filenames(folder_path):
        """
        Return a list of all files in a folder

        :param: folder_path the path to the files

        :return: list of all files in folder_path
        """
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
        return file_list

    def deconstruct_filename(image_filename):
        """Deconstructs a fully qualified filename into its components.

        Args:
            image_filename (str): The fully qualified filename to deconstruct.

        Returns:
            tuple: A tuple containing the following components:
                - directory: The directory path.
                - basename: The base filename without the extension.
                - extension: The file extension.
        """

        folder_path, file_with_extension = os.path.split(image_filename)
        file_name, file_extension = os.path.splitext(file_with_extension)

        return folder_path, file_name, file_extension

    input_files = filenames(args.input_folder)  # command line parameter processing
    file_dir, _, _ = deconstruct_filename(input_files[0])

    result_folder = None
    try:
        os.chmod(file_dir, 0o755)          # grant permission to create sub-folder
        result_folder = f"{file_dir}/result"    # synthesize the sub-folder name
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, mode=0o755, exist_ok=True)
    except OSError as e:
        print(f"Could not create the results sub-folder {result_folder}: error: {e}")

    yolo_models = ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]  # available models

    for selection in yolo_models:
        model_name = f"{selection}.pt"          # if -m | --model is unused on command line then args.model="n"
        model = YOLO(model_name)                # using an Ultralytics pre-trained model for YOLO11

        results = model(input_files)            # perform detection and prediction
        for result in results:                  # Process results list
            boxes = result.boxes                # object containing the detection bounding boxes
            """
                boxes.id  2: 384x640 2 persons, 8 birds, 321.2ms
                boxes.cls
                boxex.conf
            """
            # print(boxes.id)                   # displayed as None
            print(boxes.cls)                    # class labels for each box, see coco-labels-2014_2017.txt
            print(boxes.conf)                   # confidence scores for each box
            masks = result.masks                # object containing the detection masks
            keypoints = result.keypoints        # object containing detected keypoints for each object
            probs = result.probs                # object containing probabilities of each class for classification task
            obb = result.obb                    # object containing oriented bounding boxes
            # result.show()                       # show annotated results to screen
            path = result.path                  # path to image file
            name_only = model_name.find(".")    # remove the ".pt" suffix from model name
            file_dir, base, _ = deconstruct_filename(path)

            file_prefix = file_dir + "/result/" + base + "_" + model_name[:name_only]

            savefile = file_prefix + ".jpg"
            # result.save(filename=savefile)      # Save annotated results to file

            savetext = file_prefix + ".txt"
            # result.save_txt(filename=savetext)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="detect and predict images in specified folder using a YOLO11 model",
        epilog="YOLO11 in action!"
    )
    parser.add_argument(                        # only one input image file, please
        "input_folder",
        type=str, default="/home/chowkidar/Pictures/test/input/birds/royalswan",
        help="folder for images for detection and prediction"
    )
    parser.add_argument(
        "-m", "--model", default="n", # n for yolo11n.pt model
        type=str,
        help="model name suffix"
    )

    my_args = parser.parse_args()
    image_predict(my_args)