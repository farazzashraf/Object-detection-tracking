# Object Detection and Tracking
 
This repository contains an implementation of object detection and tracking using YOLO11 from Ultralytics integrated with DeepSORT for multi-object tracking. The project demonstrates real-time object detection and tracking capabilities by fine-tuning the YOLO model on the Mall Dataset and integrating it with DeepSORT for robust tracking.

## Features
 Object Detection: Using YOLO11 for high-accuracy detection.
 Object Tracking: DeepSORT for tracking detected objects across frames.
 Fine-Tuning: YOLO11 fine-tuned on the Mall Dataset for improved pedestrian detection.
 Real-Time Performance: Efficient integration of detection and tracking for seamless operation.

## Workflow
### 1.YOLO11 Model
 YOLO11 from Ultralytics serves as the base detection model.
 Fine-tuned using the Mall Dataset for pedestrian-specific detection.
### 2. DeepSORT Integration
 DeepSORT is used for tracking detected objects across video frames.
 YOLO11 is integrated into the detect.py script in the DeepSORT folder to serve as the detection engine.
### 3. Detection and Tracking
 Objects are first detected using the fine-tuned YOLO11 model.
 Outputs (bounding boxes and confidence scores) are passed to DeepSORT for object tracking.

## Acknowledgments
 Ultralytics YOLO11 for the detection model. - https://github.com/ultralytics/ultralytics
 DeepSORT for the tracking framework. - https://github.com/nwojke/deep_sort
