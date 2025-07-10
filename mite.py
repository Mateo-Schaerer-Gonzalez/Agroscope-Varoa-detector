import numpy as np
import cv2
import skimage
import os
import cv2
from sklearn.cluster import DBSCAN


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
        self.bbox = yolo_bbox
        self.center = ((yolo_bbox[0] + yolo_bbox[2]) // 2, (yolo_bbox[1] + yolo_bbox[3]) // 2)  # (x_center, y_center)
        self.threshold = 0.5
        self.roi_series = self.add_ROI(frames)
        self.variability = None


   
  
    def add_ROI(self, frames):
        """
        Extract ROI from all frames and append to self.roi_series.
        
        Parameters:
            frames (np.ndarray): A 4D tensor of shape (num_frames, H, W, C)
        """
        x1, y1, x2, y2 = self.bbox  # Bounding box coordinates
        roi = frames[:, y1:y2, x1:x2, :]  # Slice all frames with the bbox
        return roi
       

    def compute_variability(self, method='std'):
        
        if self.roi_series is None:
            raise ValueError("ROI series is not set. Call add_ROI() first.")

        # Convert to grayscale if needed
        roi_gray = np.mean(self.roi_series, axis=-1)  # Average across color channels

        return np.var(roi_gray, axis=0).mean() #pixel variace across frames

    def isAlive(self, method='std'):
        self.variability = self.compute_variability(method=method)
        self.alive = self.variability > self.threshold #above threshold is alive, below is dead
        return self.alive



# get mites from bbox text files per image
def get_mites_from_bboxes(result, frames):
    mites = []
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # numpy array
    for box in boxes:
        mites.append(Mite(box, frames))
    
    return mites


def draw_mite_boxes(image, mites, thickness=2, show=True, save_path=None):
    """
    Draw bounding boxes of mites on the image.

    Parameters:
        image_path (str): Path to the input image.
        mites (list of Mite): List of Mite objects with .bbox attribute (x, y, w, h).
        color (tuple): Box color in BGR (default green).
        thickness (int): Box line thickness.
        show (bool): Whether to display the image with cv2.imshow.
        save_path (str or None): If provided, saves the image to this path.

    Returns:
        image_with_boxes: The image with bounding boxes drawn.
    """

    for mite in mites:
        x1, y1, x2, y2 = mite.bbox
        if mite.isAlive():
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
       
        cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)

    if save_path:
        cv2.imwrite(save_path, image)

    return image
