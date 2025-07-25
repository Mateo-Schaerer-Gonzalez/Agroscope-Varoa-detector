import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast

# Initialize a char-level tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train the tokenizer
trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train(["code_vocab.txt"], trainer)

# Wrap it for HuggingFace compatibility
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Save
hf_tokenizer.save_pretrained("char_tokenizer")






class TextReader:
    def __init__(self, model_name="microsoft/trocr-large-handwritten", device=None):
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(model_name)  # Only image processor
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Set tokenizer IDs
        self.model.config.decoder_start_token_id = hf_tokenizer.convert_tokens_to_ids("[PAD]")
        self.model.config.pad_token_id = hf_tokenizer.convert_tokens_to_ids("[PAD]")

        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def read(self, image):
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=64)

        generated_text = hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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