from __future__ import division, print_function, absolute_import
import argparse
import os
import cv2
import torch
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from application_util import visualization
from ultralytics import YOLO

# Load YOLO model (adjust with path to your fine-tuned model)
# Initialize YOLO model
yolo_model = YOLO("yolo11n.pt")  # Load pretrained YOLO model
# yolo_model = YOLO("C:/Users/FARAZ/Desktop/Projects/Computer-Vision/deep_sort/best.pt")  # Load pretrained YOLO model

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections."""
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    detections = np.load(detection_file) if detection_file else None
    groundtruth = np.loadtxt(groundtruth_file, delimiter=',') if os.path.exists(groundtruth_file) else None
    
    image = cv2.imread(next(iter(image_filenames.values())), cv2.IMREAD_GRAYSCALE)
    image_size = image.shape if image is not None else None
    min_frame_idx, max_frame_idx = min(image_filenames.keys()), max(image_filenames.keys())
    
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": detections.shape[1] - 10 if detections is not None else 0
    }
    return seq_info


def create_detections_from_yolo(frame, min_height=0):
    """Detect only 'person' objects using YOLO model and create detections."""
    results = yolo_model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()  # Get bounding box coordinates
            conf = float(box.conf.cpu().numpy())  # Convert confidence to scalar
            cls = int(box.cls.cpu().numpy())  # Get class label as integer
            
            if cls == 0 and conf >= 0.3:  # Filter for 'person' class and confidence
                if xyxy.shape[0] == 1:  # If single bounding box
                    x1, y1, x2, y2 = map(int, xyxy[0])
                else:  # Handle multiple bounding boxes
                    x1, y1, x2, y2 = map(int, xyxy.flatten())

                bbox = [x1, y1, x2 - x1, y2 - y1]
                feature = np.array(conf)
                if bbox[3] >= min_height:
                    detections.append(Detection(bbox, feature, conf))
    return detections


# Remove the argument for --detection_file in the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT with YOLO model")
    parser.add_argument("--sequence_dir", help="Path to MOTChallenge sequence directory", default=None, required=True)
    parser.add_argument("--output_file", help="Path to the tracking output file", default="output/hypotheses2.txt")
    parser.add_argument("--min_confidence", help="Detection confidence threshold", default=0.8, type=float)
    parser.add_argument("--min_detection_height", help="Threshold on detection height", default=0, type=int)
    parser.add_argument("--nms_max_overlap", help="Non-maxima suppression threshold", default=1.0, type=float)
    parser.add_argument("--max_cosine_distance", help="Cosine distance gating threshold", default=0.2, type=float)
    parser.add_argument("--nn_budget", help="Maximum size of appearance descriptors gallery", type=int, default=100)
    parser.add_argument("--display", help="Show intermediate tracking results", default=True, type=bool)
    # parser.add_argument("--use_webcam", help="Use webcam for real-time input", action="store_true")
    return parser.parse_args()

def run(sequence_dir, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display):
    seq_info = gather_sequence_info(sequence_dir, detection_file=None)  # No detection file passed
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print(f"Processing frame {frame_idx:05d}")
        frame = cv2.imread(seq_info["image_filenames"][frame_idx])
        detections = create_detections_from_yolo(frame, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]
        
        # NMS, Tracker update, Visualization code...
        tracker.predict()
        tracker.update(detections)

        # Visualization (optional) and results
        if display:
            vis.set_image(frame.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Save results
    with open(output_file, 'w') as f:
        for row in results:
            f.write(f'{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1\n')



# def run(use_webcam, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display):
#     # Initialize the webcam
#     cap = cv2.VideoCapture(0 if use_webcam else seq_dir)

#     metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#     tracker = Tracker(metric)
#     results = []

#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()

#         # Create detections from the frame
#         detections = create_detections_from_yolo(frame, min_detection_height)
#         detections = [d for d in detections if d.confidence >= min_confidence]

#         # NMS, Tracker update, Visualization code
#         tracker.predict()
#         tracker.update(detections)
        
#         dummy_seq_info = {
#             "image_size": (640, 480),
#             "sequence_name": "webcam",
#             "min_frame_idx": 0,
#             "max_frame_idx": 1000
#         }

#         # Visualization (optional) and results
#         if display:
#             vis = visualization.Visualization(dummy_seq_info, update_ms=5)
#             vis.set_image(frame.copy())
#             vis.draw_detections(detections)
#             vis.draw_trackers(tracker.tracks)
            
#         # After updating the tracker and before showing the frame
#         for detection in detections:
#             # Draw the bounding box for the detected object
#             x1, y1, w, h = detection.to_tlbr()
#             x2, y2 = x1 + w, y1 + h
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box


#         for track in tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             bbox = track.to_tlwh()
#             results.append([len(results), track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

#         # Display the frame (optional)
#         if display:
#             cv2.imshow('Object Tracking', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     # Release the webcam and close all windows
#     cap.release()
#     cv2.destroyAllWindows()

#     return results



if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.output_file,
        # args.use_webcam, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display
    )

