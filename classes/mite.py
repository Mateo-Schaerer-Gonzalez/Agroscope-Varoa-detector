import numpy as np
import cv2
import cv2
from classes.Rect import Rect
import csv
import os

class Mite:
    def __init__(self, yolo_bbox, frames, threshold=5.0):
        """
        Initialize a Mite object with YOLO bbox and image shape.

        Parameters:pi
            yolo_bbox (tuple): (x_center, y_center, width, height)
            image_shape (tuple): (height, width) of the image
            frames (np.ndarray): A 4D tensor of shape (num_frames, H, W, C)
            threshold (float): Variability threshold to classify as alive or dead
        """
        self.bbox = Rect(*yolo_bbox)  # Create a Rect object for the bounding box
        self.center = ((yolo_bbox[0] + yolo_bbox[2]) // 2, (yolo_bbox[1] + yolo_bbox[3]) // 2)  # (x_center, y_center)
        self.threshold = 0.5
        self.roi_series = self.bbox.get_ROI(frames)
        self.variability = None
        self.assigned_rect = None
        self.alive = False
        self.max_var = None
        self.max_diff = 0
        self.cfd = 0

       

    def checkAlive(self):
        
        if self.roi_series is None:
            raise ValueError("ROI series is not set. Call add_ROI() first.")

        # Convert to grayscale if needed
        roi_gray = np.mean(self.roi_series, axis=-1)  # Average across color channels

        # Get pixel-wise range across the stack
        pixel_diff = self.roi_series.max(axis=0) - self.roi_series.min(axis=0)

        # Get the variance of these pixel-wise differences
        self.max_var = np.var(pixel_diff)

        self.max_diff = np.max(pixel_diff)

        self.variability = np.var(roi_gray, axis=0).mean()


        # Step 1: Frame-to-frame absolute difference
        frame_diffs = np.abs(np.diff(roi_gray, axis=0))  # shape: (T-1, H, W)

        T, H, W = roi_gray.shape
        self.cfd = np.sum(frame_diffs) / ((T - 1) * H * W)


        # update alive status based on variability
        # self.alive = self.variability > self.threshold #above threshold is alive, below is dead
        self.bbox.color = (0, 255, 0) if self.alive else (0, 0, 255)

        #save the data
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "variabilites.csv")
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)

         

            # Write the new row
            writer.writerow([self.alive, self.variability, self.max_var, self.max_diff, self.cfd])
                        
         


        return self.alive, self.variability #pixel variace across frames
    
    def draw(self, image, thickness=2, label=None):
        """
        Draw this mite's bounding box on the given image.
        """
        self.bbox.draw(image, thickness=thickness, label=label)

