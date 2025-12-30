import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default to printed models; switch to "handwritten" if needed.
MODEL_MODE = "printed"
MODEL_IDS = {
    "printed": ("microsoft/trocr-base-printed", "microsoft/trocr-large-printed"),
    "handwritten": ("microsoft/trocr-base-handwritten", "microsoft/trocr-large-handwritten"),
}

model_ids = MODEL_IDS.get(MODEL_MODE)
if model_ids is None:
    raise ValueError(f"Unsupported MODEL_MODE: {MODEL_MODE}")

processor_base = TrOCRProcessor.from_pretrained(model_ids[0])
processor_large = TrOCRProcessor.from_pretrained(model_ids[1])

model_base = VisionEncoderDecoderModel.from_pretrained(model_ids[0]).to(device)
model_large = VisionEncoderDecoderModel.from_pretrained(model_ids[1]).to(device)

model_base.eval()
model_large.eval()

MAX_LENGTH = 128

def predict_trocr_both(image):
    # Accept cv2 image (BGR) or PIL
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("predict_trocr_both expects np.ndarray or PIL.Image")

    image = image.convert("RGB")

    inputs_base = processor_base(images=image, return_tensors="pt")
    inputs_large = processor_large(images=image, return_tensors="pt")

    # Move tensors to device
    inputs_base = {k: v.to(device) for k, v in inputs_base.items()}
    inputs_large = {k: v.to(device) for k, v in inputs_large.items()}

    with torch.inference_mode():
        ids_base = model_base.generate(**inputs_base, max_length=MAX_LENGTH)
        ids_large = model_large.generate(**inputs_large, max_length=MAX_LENGTH)

    text_base = processor_base.batch_decode(ids_base, skip_special_tokens=True)[0]
    text_large = processor_large.batch_decode(ids_large, skip_special_tokens=True)[0]

    return {"base": text_base, "large": text_large}
