import logging

import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from helpers import log_exceptions
from global_init import device
from region import Region

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    logger.info("loading region proposal network...")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 10
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/rpn_R_50_FPN_1x.yaml")
    cfg.MODEL.DEVICE = device.type
    rpn = DefaultPredictor(cfg)
    logger.info("region proposal network is ready")


def predict_regions(image: np.ndarray):
    """
    Predict regions with objects using Region Proposal Network
    :param image: numpy ndarray, for example after cv2.imread(image_filename)
    :return: array of bounding boxes (proposed regions),
                Region([l, u, r, d]) format
    """
    logger.info("proposing regions...")
    outputs = rpn(image)
    logger.info("proposing regions done!")
    return [Region(*box.cpu().numpy()) for box in
            outputs["proposals"].proposal_boxes]


def iou(region1: Region, region2: Region):
    intersection = region1.intersect(region2)
    union_area = region1.area() + region2.area() - intersection.area()
    return intersection.area() / union_area


def filter_regions(regions, threshold=0.7):
    filtered = []
    for region in regions:
        if all(iou(region, prev_region) <= threshold
               for prev_region in filtered):
            filtered.append(region)
    return filtered
