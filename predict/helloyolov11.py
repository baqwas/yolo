#!/usr/bin/env python3
'''
    helloyolov11.py
    Simple predict run for YOLOv11 model
    2024-10-09 0.1 armw Initial DRAFT

    In the spirit of a Hello World exercise, the test image used in this
    script is from the Ultralytics website. One can replace this reference
    with any other preferred image if appropriate.
    The nano model, yolo11n.pt, is downloaded from the Ultralytics website
    if it is not present in the working folder. Also, note the minor change
    in nomenclature for the model name; there is no 'v' in the name!
    Don't forget to place an entry for the model filetype extension
    in the .gitignore file so that the model file is not uploaded
    to your repository unnecessarily.

    Reference
    https://docs.ultralytics.com/tasks/detect/#predict
'''
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # the nano model by Ultralytics
results = model("https://ultralytics.com/images/bus.jpg")  # a la Space Shuttle for AutoCAD