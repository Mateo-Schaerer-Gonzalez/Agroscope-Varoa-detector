import numpy as np
import cv2
import cv2
from classes.Rect import TextZone
import csv
import os
import pandas as pd

def getThreshold():
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "best_threshold.txt")


        with open(filename, "r") as f:
            threshold = float(f.readline().strip())

        return threshold

class Mite:
    id_counter = 0
    threshold = getThreshold()
    local_radius = 5
     
    def __init__(self, yolo_bbox, frames):
        
        
        self.mite_id = f"mite_{Mite.id_counter:04d}"
        Mite.id_counter += 1

        self.bbox = TextZone(*yolo_bbox, text = str(Mite.id_counter))  # Create a Rect object for the bounding box
        self.center = ((yolo_bbox[0] + yolo_bbox[2]) // 2, (yolo_bbox[1] + yolo_bbox[3]) // 2)  # (x_center, y_center)
        
        self.roi_series = self.bbox.get_ROI(frames)
        self.assigned_rect = None
        self.alive = True
        self.local_avg_diff = 0
        self.max_diff  = 0

  
    def to_dict(self, recording_count):
        return {'mite ID': self.mite_id,
                'zone ID': self.assigned_rect.zone_id,
                'status': 'alive' if self.alive else 'dead',
                'max diff': self.max_diff,
                'local diff': self.local_avg_diff,
                'recording': recording_count}
        

    def update_ROI(self, frames):
        self.roi_series = self.bbox.get_ROI(frames)



    def update_status(self):
        
        if self.roi_series is None:
            raise ValueError("ROI series is not set. Call add_ROI() first.")

        # Get pixel-wise range across the stack
        pixel_diff_grey = np.mean(self.roi_series.max(axis=0) - self.roi_series.min(axis=0), axis = -1)


        max_coord = np.unravel_index(np.argmax(pixel_diff_grey), pixel_diff_grey.shape)
        x, y = max_coord


        x_min = max(x - Mite.local_radius, 0)
        x_max = min(x + Mite.local_radius + 1, pixel_diff_grey.shape[0])
        y_min = max(y - Mite.local_radius, 0)
        y_max = min(y + Mite.local_radius + 1, pixel_diff_grey.shape[1])

        local_patch = pixel_diff_grey[x_min:x_max, y_min:y_max]

        # Get top 3 pixel values in the patch
        top_3_vals = np.sort(local_patch.flatten())[-3:]
        self.local_avg_diff = np.mean(top_3_vals)


        self.max_diff = np.max(pixel_diff_grey)

        # update alive status based on variability
        self.alive = 0.8 * self.max_diff + 0.2 * self.local_avg_diff > Mite.threshold #above threshold is alive, below is dead
        self.bbox.color = (0, 255, 0) if self.alive else (0, 0, 255)

       
        return self.alive
    
    

    def save_with_ground_truth(self, Ground_Truth):
        #save the data given a ground truth
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "variabilites.csv")

        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the new row
            writer.writerow([Ground_Truth, f"{'alive' if self.alive else 'dead'}", self.max_diff, self.local_avg_diff])

                
    def draw(self, image):
        """
        Draw this mite's bounding box on the given image.
        """
        self.bbox.draw(image, thickness=0.1, font_scale=0.1)

