#!/usr/bin/env python3
"""
@brief count_region.py count objects in user defined regions
@version 0.1
@date 2025-01-20
@author armw

@brief This script will infer objects in the input source using YOLO11

Copyright (C) 2025 ParkCircus Productions; All Rights Reserved.

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
count_region.py

Adapted from
@sa https://docs.ultralytics.com/guides/object-counting/#advantages-of-object-counting
Processing
        Define Region of Interest
        Detect objects in entire image/frame
        Filter object within ROI
        Output detections

Notes
for train demonstration videos:
COCO labels: {3: "car", 7: "train", 8: "truck"}

@sa https://docs.ultralytics.com/modes/predict/#why-use-ultralytics-yolo-for-inference
@sa https://docs.voxel51.com/integrations/ultralytics.html
@sa https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt
@sa https://docs.ultralytics.com/reference/solutions/object_counter/

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
import cv2
from ultralytics import YOLO, solutions

                                                    # Define region points; will need tweaking!
# region_points = [(535, 285), (1130, 285), (1130, 370), (535, 370)] # for luggage1.mp4
# region_points = [(540, 285), (1150, 285), (1150, 370), (540, 370)] # for luggage2.mp4
# region_points = [(540, 190), (1070, 190), (1070, 340), (540, 340)] # for luggage3.mp4
region_points = [(100, 370), (800, 370), (800, 555), (100, 555)] # for train_sh78_02.mp4
frame_count = 0
# classes2count = [28]                                # 28=suitcase, no 0=person & 13=bench, please
classes2count = [7]                                 # 7=train

model_name = "yolo11n.pt"                           # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}")                       # the nano model by Ultralytics
video_source = "../videos/traffic/train_sh78_02.mp4"  # input source for inference
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
counter = solutions.ObjectCounter(
    model=model_name,                               # str, path to YOLO Model file
    region=region_points,                           # list of points defining the counting region
    show=True,                                      # display the video stream
    show_in=True, show_out=True,                    # display inward & outward counts on video stream
    line_width=2)                                   # line thickness of bounding boxes

"""
counter.store_classwise_counts([classes2count[0])   # classes2count =  class index for classwise count updates
"""
while myCapture.isOpened():                         # process if frames exist
    success, frame_current = myCapture.read()       # frame-by-frame processing
    if success:                                     # frame was read
        frame_count += 1                            # keep track of the number of frames read
        tracks = model.track(frame_current,
                             persist=True,          # persist tracker if it exists already
                             show=True)
        frame_current = counter.count(frame_current)    # process data (frames | object tracks) & update object counts
        video_writer.write(frame_current)           # write the next frame to the specified video file; returns true if successful

        if cv2.waitKey(1) & 0xFF == ord('q'):       # wait for a ms for the letter 'q'
            break
    else:
        print(f"{frame_count} frames from the {video_source} were processed.")
        break

print(f"  into region of interest: {counter.in_count}") # objects moving inward
print(f"out of region of interest: {counter.out_count}")    # objects moving outward
#  print(f"counted_ids:{counter.counted_ids}")         # is this useful?
print(f"classwise_counts:\n{counter.classwise_counts}") # counts categorized by object class

myCapture.release()                                 # release video capture resources
video_writer.release()                              # release the VideoWriter object
cv2.destroyAllWindows()                             # Good Housekeeping
print("All done!")
