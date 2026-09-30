#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: hello_yolo12.py
@brief: A simple script to test YOLOv12 detection with a sample image.
@version: 1.0
@date: 2024-01-01
@license: Apache License 2.0

 Note: Ensure that the 'yolo12n.pt' model and 'coco8.yaml' dataset are available in the specified paths.
 The 'bus.jpg' image should also be present in the specified path for inference.
 You can adjust the paths as necessary for your environment.
 Note: The above code assumes that the YOLOv12 model and the COCO dataset are correctly set up in your environment.
 Make sure to install the required dependencies, such as 'ultralytics', before running this script.
 You can install it using pip:
 pip install ultralytics
 Ensure that the 'ultralytics' package is installed in your Python environment.
 You can also use a different model or dataset by changing the model path and dataset configuration.
 For more information on YOLOv12 and its usage, refer to the official documentation at:
 https://docs.ultralytics.com/
 This script is a basic example and can be extended for more complex use cases, such as
 training on custom datasets, adjusting model parameters, or integrating with other systems.
 Make sure to handle any exceptions or errors that may arise during model loading, training, or inference.
 For example, you can wrap the model loading and inference in try-except blocks to catch potential issues.
 This script is intended for educational purposes and may require modifications based on your specific setup.
 For more advanced usage, consider exploring the YOLOv12 API and its features, such as
 custom training loops, model evaluation, and deployment options.
 Ensure that you have the necessary permissions to use the YOLOv12 model and the COCO dataset.
 You can also explore other models available in the YOLOv12 family by changing the model path.
 For example, you can use 'yolo12s.pt' for a smaller model or 'yolo12l.pt' for a larger model.
 Make sure to adjust the image size (imgsz) and other parameters based on your requirements.
 This script is a starting point for working with YOLOv12 and can be customized further
 to suit your needs. You can also explore the Ultralytics GitHub repository for more
 examples and resources related to YOLOv12 and other computer vision tasks.
 For more information on the Ultralytics YOLOv12 implementation, refer to the official GitHub repository:

Source      Example                     Type            Notes
image       'image.jpg'                 str or Path 	Single image file.
URL         'https://ultralytics.com/images/bus.jpg' 	str 	URL to an image.
screenshot  'screen'                    str         Capture a screenshot.
PIL         Image.open('image.jpg') 	PIL.Image 	HWC format with RGB channels.
OpenCV      cv2.imread('image.jpg') 	np.ndarray 	HWC format with BGR channels uint8 (0-255).
numpy       np.zeros((640,1280,3)) 	np.ndarray 	HWC format with BGR channels uint8 (0-255).
torch       torch.zeros(16,3,320,640)   torch.Tensor 	BCHW format with RGB channels float32 (0.0-1.0).
CSV         'sources.csv'               str or Path 	CSV file containing paths to images, videos, or directories.
video ✅ 	'video.mp4'                 str or Path 	Video file in formats like MP4, AVI, etc.
directory   'path/'                     str or Path 	Path to a directory containing images or videos.
glob        'path/*.jpg'                str             Glob pattern to match multiple files. Use the * character as a wildcard.
YouTube ✅ 	'https://youtu.be/LNwODJXcvt4' 	str         URL to a YouTube video.
stream ✅ 	'rtsp://example.com/media.mp4' 	str         URL for streaming protocols such as RTSP, RTMP, TCP, or an IP address.
multi-stream    'list.streams'          str or Path     *.streams text file with one stream URL per row, i.e. 8 streams will run at batch-size 8.
webcam      0                           int             Index of the connected camera device to run inference on.
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model("../../images/queue.jpg")
# Access the results
for result in results:
    xywh = result.boxes.xywh    # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy    # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf   # confidence score of each box
    result.show()               # display the image with bounding boxes
