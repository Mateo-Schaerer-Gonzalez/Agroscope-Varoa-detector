import os
import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model_path = "varrodetector/model/weights/best.pt"
        self.model = YOLO(self.model_path, verbose=False)

    def run_detection(self, image_folder=None):
        if image_folder is None:
            raise ValueError("Please provide an image folder path.")

        output_folder = "runs/detect_mateo/output"
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.dng')):
                image_path = os.path.join(image_folder, filename)

                results = self.model(
                    image_path,
                    imgsz=6016,  # or (6016, 6016)
                    max_det=2000,
                    conf=0.1,
                    iou=0.5,
                    save=True,
                    show_labels=False,
                    line_width=2,
                    save_txt=True,
                    save_conf=True,
                    project=os.path.dirname(output_folder),
                    name=os.path.basename(output_folder),
                    verbose=False,
                    batch=1,
                    exist_ok=True,
                )

                for result in results:
                    print(f"🔍 {filename} detections:")
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].tolist()
                        conf = boxes.conf[i].item()
                        cls = int(boxes.cls[i].item())
                        print(f"  Class {cls}, Confidence {conf:.2f}, Box {xyxy}")

if __name__ == "__main__":
    detector = Detector()
    detector.run_detection(image_folder="Sample images")
