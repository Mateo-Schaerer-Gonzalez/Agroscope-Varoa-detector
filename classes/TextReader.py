import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2

class TextReader:
    def __init__(self, model_name="microsoft/trocr-large-handwritten", device=None):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def read(self,image):
        """
        There are 3 main models to choose from, small, base and large. 
        Some other fine-tuned models: IAM Handwritten, SROIE Receipts
        """

        # Check for GPU availability

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, 
                                            max_new_tokens=30)
        
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        return generated_text.strip()



def has_text(img: np.ndarray, threshold: float = 0.02) -> bool:
    """
    Check if the image likely contains text by detecting structured edges.

    Args:
        img: np.ndarray, RGB image (H x W x 3), dtype=uint8.
        threshold: float, minimum edge density to consider as text.

    Returns:
        bool: True if text likely present, False otherwise.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use Canny edge detector to find stroke-like edges
    edges = cv2.Canny(gray, 50, 150)

    # Ratio of edge pixels to total image size
    edge_ratio = np.sum(edges > 0) / edges.size
    

    return edge_ratio > threshold
