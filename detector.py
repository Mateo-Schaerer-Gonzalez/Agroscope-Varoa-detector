import os
import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model_path = "varrodetector/model/weights/best.pt"
        self.model = YOLO(self.model_path, verbose=False)
        self.result = None

    def run_detection(self, image):
        # Run the model on a single image
        result = self.model(
            image,
            imgsz=6016, #was 6012
            max_det=2000,
            conf=0.7,
            iou=0.5,
            save=False,
            show_labels=False,
            line_width=2,
            save_txt=False,
            save_conf=False,
            verbose=False,
            batch=1,
            exist_ok=True,
        )

        self.result = result[0] # only for one image
