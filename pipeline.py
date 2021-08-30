import logging

import numpy as np
import PIL

from region_proposal import predict_regions
from classifier import get_categories_probs
from text_processor import find_categories_in_text

logger = logging.getLogger(__name__)


def pil_to_cv2(image: PIL.Image):
    return np.array(image.convert('RGB'))


def search_on_image(image: PIL.Image, text: str):
    regions = predict_regions(pil_to_cv2(image))
    detections = []
    for region in regions:
        cropped = image.crop(box=region)
        probs = get_categories_probs(cropped)
        query_en, query_ru = find_categories_in_text(text)
        detections += [(cat, float(probs[cat]), *region) for cat in
                       np.concatenate((query_en, query_ru)) if
                       probs[cat] > 0.2]
