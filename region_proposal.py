import logging

import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from helpers import log_exceptions
from global_init import device

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/rpn_R_50_FPN_1x.yaml")
    cfg.MODEL.DEVICE = device.type
    rpn = DefaultPredictor(cfg)


def predict_regions(image: np.ndarray):
    """
    Predict regions with objects using Region Proposal Network
    :param image: numpy ndarray, for example after cv2.imread(image_filename)
    :return: array of bounding boxes (proposed regions), [l, u, r, d] format
    """
    outputs = rpn(image)
    return [box.cpu().numpy() for box in outputs["proposals"].proposal_boxes]
