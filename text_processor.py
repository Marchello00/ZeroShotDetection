import logging

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

from helpers import log_exceptions
from classifier import imagenet_classes

logger = logging.getLogger(__name__)

with log_exceptions(logger):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()

    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")
    stopwords = russian_stopwords + english_stopwords


def normalize_text(text):
    tokens = mystem.lemmatize(text.lower())  # for russian
    # additional lemmatize for english
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens \
                       if token not in stopwords \
                       and token != " " \
                       and token.strip() not in punctuation]
    return " " + " ".join(filtered_tokens) + " "  # for easier searching


with log_exceptions(logger):
    rus_categories_normalized = [normalize_text(label) for label in
                                imagenet_classes["standart_en"]]
    categories_normalized = [normalize_text(label) for label in
                             imagenet_classes["standart_ru"]]

    rus_coarse_normalized = imagenet_classes["groups_ru_norm"]
    en_coarse_normalized = imagenet_classes["groups_en_norm"]


def find_in_text(text, categories):
    return np.array([i for i, label in enumerate(categories) if label in text],
                    dtype=np.int32)


def expand_groups(text, coarse):
    for category, items in coarse.items():
        if not category.strip():
            continue
        if category in text:
            logger.info(f"found category {category}")
            text += items
    return text


def find_categories_in_text(text):
    text = normalize_text(text)
    logger.info(f"normalized text: {text}")
    text = expand_groups(text, en_coarse_normalized)
    text = expand_groups(text, rus_coarse_normalized)
    return find_in_text(text, categories_normalized), \
           find_in_text(text, rus_categories_normalized)
