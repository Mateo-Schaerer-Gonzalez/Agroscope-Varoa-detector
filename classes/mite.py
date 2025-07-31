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
        self.threshold = 20
        self.roi_series = self.bbox.get_ROI(frames)
        self.assigned_rect = None
        self.alive = True
        self.local_avg_diff = 0
        self.max_diff  = 0
        self.local_radius = 5

       

    def checkAlive(self):
        
        if self.roi_series is None:
            raise ValueError("ROI series is not set. Call add_ROI() first.")

                # Step 1: Convert to grayscale if needed
        roi_gray = np.mean(self.roi_series, axis=-1)  # shape: (T, H, W)

        # Step 2: Compute absolute frame-to-frame differences
        frame_diffs = np.abs(np.diff(roi_gray, axis=0))  # shape: (T-1, H, W)

    
        # Get pixel-wise range across the stack
        pixel_diff = np.mean(self.roi_series.max(axis=0) - self.roi_series.min(axis=0), axis = -1)


       


        # Step 2: Find pixel with maximum range
        max_coord = np.unravel_index(np.argmax(pixel_diff), pixel_diff.shape)
        x, y = max_coord


        x_min = max(x - self.local_radius, 0)
        x_max = min(x + self.local_radius + 1, pixel_diff.shape[0])
        y_min = max(y - self.local_radius, 0)
        y_max = min(y + self.local_radius + 1, pixel_diff.shape[1])

        local_patch = pixel_diff[x_min:x_max, y_min:y_max]

        # Step 4: Get top 3 pixel values in the patch
        top_3_vals = np.sort(local_patch.flatten())[-3:]
        self.local_avg_diff = np.mean(top_3_vals)


        self.max_diff = np.max(pixel_diff)



        # update alive status based on variability
        self.alive = 0.8 * self.max_diff + 0.2 * self.local_avg_diff > self.threshold #above threshold is alive, below is dead
        self.bbox.color = (0, 255, 0) if self.alive else (0, 0, 255)

        #save the data
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "variabilites.csv")
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)

         

            # Write the new row
            writer.writerow([self.alive, self.max_diff, self.local_avg_diff])
                        
         


        return self.alive, self.local_avg_diff #pixel variace across frames
    
    def draw(self, image, thickness=2, label=None):
        """
        Draw this mite's bounding box on the given image.
        """
        self.bbox.draw(image, thickness=thickness, label=label)

