import os
import logging

import torch

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from helpers import log_exceptions

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(0)  # singe gpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"zero-shot object detection is set to run on {device}")
