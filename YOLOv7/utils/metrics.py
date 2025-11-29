import subprocess
import os
import json
import torch
from pathlib import Path
from ultralytics import YOLO

# Custom class names for YOLOv5 and YOLOv7
CUSTOM_CLASS_NAMES = ["PL", "BX", "TJ", "CK", "ZAW", "CJ"]

# Class names for YOLOv8
YOLOV8_CLASS_NAMES = ['Deformation', 'Obstacle', 'Rupture', 'Disconnect', 'Misalignment', 'Deposition']

# Function to map class IDs to custom names for YOLOv5 and YOLOv7
def map_class_ids_to_custom_names(results, class_names=CUSTOM_CLASS_NAMES):
    mapped_results = []
    for result in results:
        class_id = result['class']
        class_name = class_names[class_id]  # Use custom class name mapping
        result['class_name'] = class_name  # Add custom class name to the result
        mapped_results.append(result)
    return mapped_results

# Function to map class IDs to class names for YOLOv8
def map_class_ids_to_yolov8_names(results, class_names=YOLOV8_CLASS_NAMES):
    mapped_results = []
    for result in results:
        class_id = result['class']
        class_name = class_names[class_id]  # Use YOLOv8 class name mapping
        result['class_name'] = class_name  # Add class name to the result
        mapped_results.append(result)
    return mapped_results

# YOLOv5 Evaluation Logic
def run_yolov5_evaluation(data_yaml, weights_path, img_size=640, batch_size=16, iou_thres=0.5):
    print("Running YOLOv5 Evaluation...")

    # Use subprocess to run YOLOv5 evaluation with the specified parameters
    yolov5_cmd = [
        "python", "D:/yolov5/val.py",  # Correct path to val.py for YOLOv5
        "--data", data_yaml, 
        "--weights", weights_path, 
        "--img-size", str(img_size),
        "--batch-size", str(batch_size),
        "--iou-thres", str(iou_thres),
        "--save-json"
    ]
    
    subprocess.run(yolov5_cmd, check=True)
    
    results_file = os.path.join('runs', 'val', 'exp', 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as file:
            results = json.load(file)
        
        # Map class IDs to custom names
        mapped_results = map_class_ids_to_custom_names(results, CUSTOM_CLASS_NAMES)
        print(f"YOLOv5 Evaluation Results:\n{mapped_results}")
    else:
        print("YOLOv5 results not found. Check if the evaluation ran correctly.")

# YOLOv7 Evaluation Logic
def run_yolov7_evaluation(data_yaml, weights_path, img_size=640, batch_size=16, iou_thres=0.5):
    print("Running YOLOv7 Evaluation...")
    # Use subprocess to run YOLOv7 evaluation with the specified parameters
    yolov7_cmd = [
        "python", "D:/yolov7/test.py",  # Correct path to test.py for YOLOv7
        "--data", data_yaml, 
        "--weights", weights_path, 
        "--img-size", str(img_size), 
        "--batch-size", str(batch_size), 
        "--iou-thres", str(iou_thres),
        "--save-json"
    ]
    
    subprocess.run(yolov7_cmd, check=True)
    
    results_file = os.path.join('runs', 'test', 'exp', 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as file:
            results = json.load(file)
        
        # Map class IDs to custom names
        mapped_results = map_class_ids_to_custom_names(results, CUSTOM_CLASS_NAMES)
        print(f"YOLOv7 Evaluation Results:\n{mapped_results}")
    else:
        print("YOLOv7 results not found. Check if the evaluation ran correctly.")

# YOLOv8 Evaluation Logic
def run_yolov8_evaluation(model_path, data_yaml, img_size=640, batch_size=16, iou_thres=0.5):
    print("Running YOLOv8 Evaluation...")
    
    # Using the official YOLOv8 library (ultralytics) to load the model
    model = YOLO(model_path)  # Load the YOLOv8 model
    results = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, iou=iou_thres, save_json=True)

    # The results will be stored in the 'results' object. We can map class IDs to YOLOv8 names.
    mapped_results = map_class_ids_to_yolov8_names(results, YOLOV8_CLASS_NAMES)
    print(f"YOLOv8 Evaluation Results:\n{mapped_results}")

# Main function to call the evaluation for all three models
def main():
    yolov5_data_yaml = "D:/yolo app/yolov5/data/coco.yaml"  # Updated path for YOLOv5
    yolov7_data_yaml = "D:/yolo app/yolov5/data/coco.yaml"  # YOLOv7 dataset path
    yolov8_data_yaml = "D:/yolov8/data/config.yaml"  # YOLOv8 dataset path

    yolov5_weights = "D:/yolo app/models/yolov5_best_latest_20230224.pt"
    yolov7_weights = "D:/pipeline_detection_system1/YOLOv7/weights/best.pt"
    yolov8_weights = "D:/yolo app/models/efficientnet.pt"

    run_yolov5_evaluation(yolov5_data_yaml, yolov5_weights)
    run_yolov7_evaluation(yolov7_data_yaml, yolov7_weights)
    run_yolov8_evaluation(yolov8_weights, yolov8_data_yaml)

if __name__ == "__main__":
    main()
