import numpy as np
import cv2
import cv2
from classes.Rect import Rect
import csv
import os

class Mite:
    def __init__(self, yolo_bbox, frames):
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
        
        self.roi_series = self.bbox.get_ROI(frames)
        self.assigned_rect = None
        self.alive = True
        self.local_avg_diff = []
        self.max_diff  = []
        self.local_radius = 5


        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "best_threshold.txt")


        with open(filename, "r") as f:
            self.threshold = float(f.readline().strip())

    def update_ROI(self, frames):
        self.roi_series = self.bbox.get_ROI(frames)



    def checkAlive(self, Ground_Truth):
        
        if self.roi_series is None:
            raise ValueError("ROI series is not set. Call add_ROI() first.")



  
        # Get pixel-wise range across the stack
        pixel_diff_grey = np.mean(self.roi_series.max(axis=0) - self.roi_series.min(axis=0), axis = -1)


       


        max_coord = np.unravel_index(np.argmax(pixel_diff_grey), pixel_diff_grey.shape)
        x, y = max_coord


        x_min = max(x - self.local_radius, 0)
        x_max = min(x + self.local_radius + 1, pixel_diff_grey.shape[0])
        y_min = max(y - self.local_radius, 0)
        y_max = min(y + self.local_radius + 1, pixel_diff_grey.shape[1])

        local_patch = pixel_diff_grey[x_min:x_max, y_min:y_max]

        # Get top 3 pixel values in the patch
        top_3_vals = np.sort(local_patch.flatten())[-3:]
        self.local_avg_diff.append(np.mean(top_3_vals))


        self.max_diff.append(np.max(pixel_diff_grey))



        # update alive status based on variability
        self.alive = 0.8 * self.max_diff[-1] + 0.2 * self.local_avg_diff[-1] > self.threshold #above threshold is alive, below is dead
        self.bbox.color = (0, 255, 0) if self.alive else (0, 0, 255)

        #save the data
    
        script_dir = os.path.dirname(os.path.abspath(__file__))

        filename = os.path.join(script_dir, "variabilites.csv")

        if Ground_Truth == "alive" or Ground_Truth =="dead":
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)

            

                # Write the new row
                writer.writerow([Ground_Truth, f"{'alive' if self.alive else 'dead'}", self.max_diff[-1], self.local_avg_diff[-1]])
                            
         

        return self.alive, self.max_diff[-1] 
    
    

    
    def draw(self, image, thickness=2, label=None):
        """
        Draw this mite's bounding box on the given image.
        """
        self.bbox.draw(image, thickness=thickness, label=label)

