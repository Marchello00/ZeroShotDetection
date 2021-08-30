import logging

import numpy as np
import torch

from classifier import get_categories_probs, imagenet_classes
from text_processor import find_categories_in_text

logger = logging.getLogger(__name__)


def search_on_image(image, text):
    probs = get_categories_probs(image)

    top5_prob, top5_catid = torch.topk(probs, 5)
    logger.info("image classifiaction top-5:")
    for i in range(top5_prob.size(0)):
        logger.info(
            f"{i}: {imagenet_classes['standart_en'][top5_catid[i]]} {top5_prob[i].item()}")

    query_en, query_ru = find_categories_in_text(text)

    return [(cat, float(probs[cat]), 0.1, 0.1, 0.9, 0.9) for cat in
            np.concatenate((query_en, query_ru)) if probs[cat] > 0.2]
