import logging
import re

from transformers import pipeline

from helpers import log_exceptions
from global_init import device

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    logger.info("preparing translator")
    translator_en_ru = pipeline(task='translation',
                                model='Helsinki-NLP/opus-mt-en-ru',
                                device=-1 if device.type == 'cpu' else 0)
    translator_ru_en = pipeline(task='translation',
                                model='Helsinki-NLP/opus-mt-ru-en',
                                device=-1 if device.type == 'cpu' else 0)
    logger.info("translator is ready")


def translate_en_ru(texts):
    return translator_en_ru(texts)


def translate_ru_en(texts):
    return translator_ru_en(texts)


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))
