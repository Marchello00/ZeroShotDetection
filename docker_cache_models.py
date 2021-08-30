import torch

from torchvision.models import inception_v3

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Inception_v3
inception_v3(pretrained=True)

# Region Proposal Network
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/rpn_R_50_FPN_1x.yaml")
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DefaultPredictor(cfg)
