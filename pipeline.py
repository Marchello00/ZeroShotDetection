import logging
from collections import defaultdict

import numpy as np
import PIL

from region_proposal import predict_regions, filter_regions
from classifier import get_categories_probs
from text_processor import find_categories_in_text

logger = logging.getLogger(__name__)


def pil_to_cv2(image: PIL.Image):
    return np.array(image.convert('RGB'))


def search_on_image(image: PIL.Image,
                    text: str,
                    classification_threshold=0.2,
                    overlap_threshold=0.7,
                    same_class_overlap_threshold=0.3):
    regions = predict_regions(pil_to_cv2(image))
    regions = filter_regions(regions, threshold=overlap_threshold)

    query = find_categories_in_text(text)
    if not query:
        return []

    detections = []
    for region in regions:
        cropped = image.crop(box=region.unwrap())
        probs = get_categories_probs(cropped)[query]
        winner_id = probs.argmax()
        if probs[winner_id] > classification_threshold:
            region.probability = probs[winner_id]
            region.idx = query[winner_id]
            detections.append(region)

    # remove everything that overlaps significantly
    # among detections of the same class
    detections_by_category = defaultdict(list)
    for detection in detections:
        detections_by_category[detection.idx].append(detection)
    detections = []
    for cat, cat_detections in detections_by_category.items():
        detections += filter_regions(cat_detections,
                                     threshold=same_class_overlap_threshold)
    return detections
