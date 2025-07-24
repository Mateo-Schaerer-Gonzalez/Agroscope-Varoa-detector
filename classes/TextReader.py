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
                                            max_new_tokens=30,
                                            num_beams=5,
                                            early_stopping=True,
                                            no_repeat_ngram_size=2)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        return generated_text.strip()


def has_text(img: np.ndarray, threshold: float = 0.01, dark_pixel_value: int = 200) -> bool:
    """
    Check if the given RGB image likely contains text based on dark pixel ratio.

    Args:
        img: np.ndarray, RGB image (H x W x 3), dtype=uint8.
        threshold: float, ratio of dark pixels below which we say no text.
        dark_pixel_value: int, pixel intensity threshold to consider as "dark" (0-255).

    Returns:
        bool: True if text likely present, False otherwise.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold to binary image: text pixels should be darker than dark_pixel_value
    _, binary = cv2.threshold(gray, dark_pixel_value, 255, cv2.THRESH_BINARY_INV)

    # Calculate ratio of dark pixels (potential text pixels)
    dark_ratio = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])

    # Debug: print(dark_ratio)

    # If dark pixels exceed threshold, we say text is present
    return dark_ratio > threshold