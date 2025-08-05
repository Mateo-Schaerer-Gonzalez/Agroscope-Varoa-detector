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
        print(f'using: {self.device}')

    def read(self, image):
        """
        There are 3 main models to choose from, small, base and large. 
        Some other fine-tuned models: IAM Handwritten, SROIE Receipts
        """

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            max_new_tokens=30,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_ids = outputs.sequences
        scores = outputs.scores  # list of logits for each generated token step

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        probs = []
        for i, logits in enumerate(scores):
            softmax_probs = torch.softmax(logits, dim=-1)
            token_id = generated_ids[0, i + 1]  # +1 because first token is input
            token_prob = softmax_probs[0, token_id].item()
            probs.append(token_prob)

        avg_confidence = sum(probs) / len(probs) if probs else 0
        CONFIDENCE_THRESHOLD = 0.7

        if avg_confidence < CONFIDENCE_THRESHOLD:
            return "EMPTY"
        else:
            print(generated_text)
            return generated_text.strip()


