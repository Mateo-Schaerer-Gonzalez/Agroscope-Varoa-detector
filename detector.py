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
            imgsz=6016, 
            max_det=2000,
            conf=0.6,
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


#load detector and fine tune it:
detector = Detector()

# Load a pretrained model (or your existing one)
model = YOLO("varrodetector/model/weights/best.pt")  # or 'varrodetector/model/weights/best.pt' if you want to continue training

# Train (fine-tune) on your dataset
model.train(
    data="data.yaml",  # path to your dataset YAML
    epochs=50,
    imgsz=6016,
    batch=8,
    name="fine_tuned_varro_model",
    resume=False  # True if you're continuing from a checkpoint
)
