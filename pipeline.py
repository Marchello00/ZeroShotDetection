import logging
from collections import defaultdict

import numpy as np
import torch
from PIL.Image import Image

from helpers import pil_to_cv2
from region_proposal import predict_regions, filter_regions
from classifier import get_categories_probs, get_cat_synset
from similarity_check import get_similarities
from translate import has_cyrillic, translate_ru_en

logger = logging.getLogger(__name__)


def search_on_image(image: Image,
                    text: str,
                    classification_threshold=0.2,
                    similarity_threshold=0.3,
                    overlap_threshold=0.3,
                    same_class_overlap_threshold=0.1,
                    n_predictions_for_region=3):
    if has_cyrillic(text):
        text = translate_ru_en([text])[0]
        logger.info(f"russian text was translated: {text}")

    regions = predict_regions(pil_to_cv2(image))
    regions = filter_regions(regions, threshold=overlap_threshold)

    detections = []
    for region in regions:
        logger.info(f"analysing region: {region.unwrap()}")
        cropped = image.crop(box=region.unwrap())
        probs = get_categories_probs(cropped)

        topn_prob, topn_catid = torch.topk(probs, n_predictions_for_region)
        filt = topn_prob > classification_threshold
        if not filt.any():
            logger.info("no matches, skipping region")
            continue
        topn_prob = topn_prob[filt]
        topn_catid = topn_catid[filt]
        logger.info("got top:")
        for i, (prob, idx) in enumerate(zip(topn_prob, topn_catid)):
            logger.info(f"{i}: {get_cat_synset(idx)[0]} {prob}")
        similarities = get_similarities(text, [', '.join(get_cat_synset(idx))
                                               for idx in topn_catid])
        logger.info(f"similarities: {similarities}")
        best_match = np.array(similarities).argmax()
        if topn_prob[best_match] > classification_threshold and \
                similarities[best_match] > similarity_threshold:
            region.probability = topn_prob[best_match]
            region.idx = topn_catid[best_match]
            logger.info(
                f"approved best match: {get_cat_synset(region.idx)[0]}")
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
