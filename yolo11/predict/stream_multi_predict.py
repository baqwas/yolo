 #!/usr/bin/env python3
"""
@brief stream_multi_predict.py run a simple Predict Mode script with a pretrained YOLO11 model
@version 0.1
@date 2024-10-20
@author armw

@brief Run inference on live video streams using RTSP, RTMP, TCP, or IP address protocols

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
stream_multi_predict.py

Using stream=True, to create a generator of Results objects to reduce memory usage for long videos
For multiple streams, a .streams text file can be used to perform batched inference, 
where the batch size is determined by the number of streams provided (e.g., batch-size 8 for 8 streams).

Example .streams text file:
rtsp://raspbari14.parkcircus.org/media.mp4
rtsp://raspbari17.parkcircus.org/media.mp4
rtmp://raspbari42.parkcircus.org/live
tcp://192.168.43.1:5554
...

Each record in the above text file represents a streaming source,
one can monitor and perform inference on several video streams concurrently.

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

model_name = "yolo11n"      # pretrained Ultralytics model for YOLO11, nano, COCO dataset
model = YOLO(f"{model_name}.pt")    # the nano model by Ultralytics

# Single stream with batch-size 1 inference
multi_stream = "../videos/streams.txt"  # RTSP, RTMP, TCP, or IP streaming address

results = model(multi_stream, stream=True)    # using a parameter driven value for input source

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
    for box in result.boxes:    # iterate through all boxes objects in the results object
        for label in box.cls:   # demonstration to check if a specific label was detected
            print(f"{result.names[int(label)]} {box.conf[0]:.2f}")
            if result.names[int(label)] == "person":  # interrogate persons
                print("Halt! Wer da?")    # example to substitute print statement with an alert function