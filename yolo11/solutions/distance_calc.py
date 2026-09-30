#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Calculation using Ultralytics Python package
Author: Matha Goram
Date: 2025-10-14
Description: This script demonstrates how to perform distance calculation using the Ultralytics Python package.
It processes a video file, detects objects using a pre-trained YOLO11 model, and calculates
the distance between detected objects.
@license: MIT
@version: 1.0
@status: Development
@repository: GitHub - baqwas
@maintainer: Matha Goram
@contact: armw
@see: https://docs.ultralytics.com/guides/distance-calculation/#how
Keywords: distance calculation, ultralytics, python, computer vision, object detection
Usage: python distance_calculation.py
Requirements: ultralytics package, opencv-python
Example: python distance_calculation.py --model path/to/yolo11/model --video path/to
    * Loads a pre-trained YOLO11 model
    * Processes a video file frame by frame
    * Detects objects and calculates distances between them
"""
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("../videos/elephant/elephant_train_yolo11n.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize distance calculation object
distancecalculator = solutions.DistanceCalculation(
    model="yolo11n.pt",  # path to the YOLO11 model file.
    show=True,  # display the output
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = distancecalculator(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows