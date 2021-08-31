from contextlib import contextmanager

import sentry_sdk
import numpy as np
from PIL.Image import Image


def pil_to_cv2(image: Image):
    return np.array(image.convert('RGB'))


@contextmanager
def log_exceptions(logger):
    try:
        yield None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception(e)
        raise e
