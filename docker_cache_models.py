import torch

from torchvision.models import inception_v3

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import nltk
from pymystem3 import Mystem

from sentence_transformers import SentenceTransformer

from transformers import pipeline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inception_v3
inception_v3(pretrained=True)

# Region Proposal Network
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/rpn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/rpn_R_50_FPN_1x.yaml")
cfg.MODEL.DEVICE = device
DefaultPredictor(cfg)

# Text processing
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
Mystem()

# Sentence transformer
SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# Translator
pipeline(task='translation', model='Helsinki-NLP/opus-mt-en-ru', device=device)
pipeline(task='translation', model='Helsinki-NLP/opus-mt-ru-en', device=device)
