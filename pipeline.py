import logging
from collections import defaultdict

from PIL.Image import Image

from region_proposal import predict_regions_compressed, filter_regions
from classifier import get_categories_probs
from text_processor import find_categories_in_text

logger = logging.getLogger(__name__)


def search_on_image(image: Image,
                    text: str,
                    classification_threshold=0.2,
                    overlap_threshold=0.7,
                    same_class_overlap_threshold=0.3,
                    region_compress_size=(300, 300)):
    query = find_categories_in_text(text)
    if not query:
        return []

    regions = predict_regions_compressed(image, size=region_compress_size)
    regions = filter_regions(regions, threshold=overlap_threshold)

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
