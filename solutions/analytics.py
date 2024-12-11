#!/usr/bin/env python3
"""
@brief analytics.py generate analytical graphs using YOLO
@version 0.1
@date 2024-10-20
@author armw

@brief This script will calculate the distance between two selected bounding boxes whose coordinates are limited to 2D only.
Be very wary about using the value returned by this script except for visualization purposes.


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
analytics.py


Adapted from
@sa hhttps://docs.ultralytics.com/guides/analytics/
Processing


Notes
Enhances accurate spatial positioning in computer vision tasks
Estimates object size for better contextual understanding

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
@sa https://docs.ultralytics.com/reference/solutions/object_counter/

Adapted from https://docs.ultralytics.com/guides/distance-calculation/#visuals
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
import cv2
from ultralytics import solutions

frame_count = 0
model_name = "yolo11n.pt"                           # pretrained Ultralytics model for YOLO11, nano, COCO dataset

video_source = "../videos/elephant/elephant_train.mov"  # input source for distance calculation between user selected bounding boxes

if not os.path.isfile(video_source):                # does the source video exist?
    print(f"Unable to read source video at {video_source}")
    exit(-1)

myCapture = cv2.VideoCapture(video_source)          # leverage OpenCV to access the source video
assert myCapture.isOpened(), f"Error reading video file {video_source}"
w, h, fps = (int(myCapture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
                                                    # Video writer
video_output: str = os.path.splitext(video_source)[0] + "_" +  \
    os.path.splitext(model_name)[0] + ".mp4"        # let's get output filename aligned to source filename
                                                    # MJPG, XVID, DIVX, H264, FFMPEG, etc.
video_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # 4-byte code for video codec
                                                    # Init Object Counter
distance = solutions.DistanceCalculation(model=model_name,  # path  to Ultralytics YOLO model file
                                         show=True,         # flag to control whether to display the video stream
                                        line_width=2)       # line thickness for bounding boxes

while myCapture.isOpened():                         # process if frames exist
    success, frame_current = myCapture.read()       # frame-by-frame processing
    if success:                                     # frame was read
        frame_count += 1
        frame_current = distance.calculate(frame_current)
        video_writer.write(frame_current)

        if cv2.waitKey(1) & 0xFF == ord('q'):       # wait for a ms for the letter 'q'
            break
    else:                                           # frame from video unavailable, let's exit
        print(f"{frame_count} frames from the {video_source} were processed.")  # nominal feedback
        break

myCapture.release()                                 # release video capture resources
video_writer.release()                              # release the VideoWriter object
cv2.destroyAllWindows()                             # Good Housekeeping
print("All done!")
