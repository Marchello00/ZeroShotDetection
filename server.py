import logging
import time
import os
import json

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from flask import Flask, request, jsonify
from healthcheck import HealthCheck
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), integrations=[FlaskIntegration()])

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)

MASK_ID = 103
try:
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(0)  # singe gpu
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"zero-shot object detection is set to run on {device}")

    # init model
    model = inception_v3(pretrained=True).to(device)
    model.eval()

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

    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()

    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")
    stopwords = russian_stopwords + english_stopwords

    logger.info("preparations are done")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")
logging.getLogger("werkzeug").setLevel("WARNING")


def get_categories_probs(input_image):
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    return probs.to('cpu')


def normalize_text(text):
    tokens = mystem.lemmatize(text.lower())  # for russian
    # additional lemmatize for english
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens \
                       if token not in stopwords \
                       and token != " " \
                       and token.strip() not in punctuation]
    return " " + " ".join(filtered_tokens) + " "  # for easier searching


def find_in_text(text, categories):
    return np.array([i for i, label in enumerate(categories) if label in text],
                    dtype=np.int)


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


def search_on_image(image, text):
    probs = get_categories_probs(image)

    top5_prob, top5_catid = torch.topk(probs, 5)
    logger.info("image classifiaction top-5:")
    for i in range(top5_prob.size(0)):
        logger.info(
            f"{i}: {imagenet_classes['standart_en'][top5_catid[i]]} {top5_prob[i].item()}")

    query_en, query_ru = find_categories_in_text(text)

    return [(cat, float(probs[cat]), 0.1, 0.1, 0.9, 0.9) for cat in \
            np.concatenate((query_en, query_ru)) if probs[cat] > 0.2]


try:
    rus_categories_normalized = [normalize_text(label) for label in
                                 imagenet_classes["standart_en"]]
    categories_normalized = [normalize_text(label) for label in
                             imagenet_classes["standart_ru"]]

    rus_coarse_normalized = imagenet_classes["groups_ru_norm"]
    en_coarse_normalized = imagenet_classes["groups_en_norm"]
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e


@app.route("/respond", methods=["POST"])
def respond():
    st_time = time.time()

    texts = request.json.get("text", [])
    img_paths = request.json.get("image", [])

    logger.info(f"got request: text={texts}, image={img_paths}")

    if len(texts) != len(img_paths):
        error = f"numbers of texts and images must be equal"
        logger.info(error)
        return jsonify({"error": error})

    results = []
    for text, img_path in zip(texts, img_paths):
        try:
            image = Image.open(img_path)
            results_id = search_on_image(image, text)
            results.append([(
                imagenet_classes["standart_en"][idx],
                imagenet_classes["standart_ru"][idx],
                prob,
                l, u, r, d
            ) for idx, prob, l, u, r, d in results_id])
        except Exception as exc:
            logger.exception(exc)
            sentry_sdk.capture_exception(exc)
            results.append([])

    total_time = time.time() - st_time
    logger.info(f"zero-shot object detection exec time: {total_time:.3f}s")
    return jsonify({"detected_objects": results})
