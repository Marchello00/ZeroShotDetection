import sentry_sdk
from contextlib import contextmanager


@contextmanager
def log_exceptions(logger):
    try:
        yield None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.exception(e)
        raise e
