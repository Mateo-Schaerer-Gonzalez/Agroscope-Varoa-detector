import numpy as np
import cv2
import skimage
import os
import cv2


class Mite:
    def __init__(self, yolo_bbox, threshold=5.0):
        """
        Initialize a Mite object with YOLO bbox and image shape.

        Parameters:
            yolo_bbox (tuple): (x_center, y_center, width, height)
            image_shape (tuple): (height, width) of the image
            threshold (float): Variability threshold to classify as alive or dead
        """
        self.bbox = yolo_bbox
        self.threshold = 0.5
        self.roi_series = []
        self.alive = True


   
  
    def add_image(self, image):
        x1, y1, x2, y2  = self.bbox
        roi = image[y1:y2, x1:x2]
        self.roi_series.append(roi)

    def compute_variability(self, method='std'):
        if not self.roi_series:
            return None

        stack = np.stack(self.roi_series, axis=0)

        if stack.ndim == 4:  # Color images
            stack_gray = np.mean(stack, axis=-1)
        else:
            stack_gray = stack

        if method == 'std':
            variability = np.std(stack_gray, axis=0)
        elif method == 'var':
            variability = np.var(stack_gray, axis=0)
        elif method == 'entropy':
            from skimage.measure import shannon_entropy
            variability = np.array([
                shannon_entropy(stack_gray[:, i, j])
                for i in range(stack_gray.shape[1])
                for j in range(stack_gray.shape[2])
            ]).reshape(stack_gray.shape[1:])
        else:
            raise ValueError("Unsupported method")

        return np.mean(variability)

    def isAlive(self, method='std'):
        #variability_score = self.compute_variability(method=method)
        #self.alive = variability_score > self.threshold
        return self.alive



# get mites from bbox text files per image
def get_mites_from_bboxes(result):
    mites = []
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # numpy array
    for box in boxes:
        mites.append(Mite(box))
    
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
