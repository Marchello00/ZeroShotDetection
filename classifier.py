import logging
import json

import torch
from torchvision.models import inception_v3
from torchvision import transforms

from global_init import device
from helpers import log_exceptions

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    classifier = inception_v3(pretrained=True).to(device)
    classifier.eval()

    logger.info('inception_v3 loaded successfully')

    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with open('imagenet_classes.json', 'r') as f:
        imagenet_classes = json.load(f)


def get_categories_probs(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    return probs.to('cpu')
