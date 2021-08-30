import logging
import time

from PIL import Image
from flask import Flask, request, jsonify
from healthcheck import HealthCheck
import sentry_sdk

from classifier import imagenet_classes
from pipeline import search_on_image

logger = logging.getLogger(__name__)

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")
logging.getLogger("werkzeug").setLevel("WARNING")


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
            results_local = search_on_image(image, text)
            results.append([
                {"label_en": imagenet_classes["standart_en"][region.idx],
                 "label_ru": imagenet_classes["standart_ru"][region.idx],
                 "l": region.l, "u": region.u, "r": region.r, "d": region.d}
                for region in results_local])
        except Exception as exc:
            logger.exception(exc)
            sentry_sdk.capture_exception(exc)
            results.append([])

    total_time = time.time() - st_time
    logger.info(f"zero-shot object detection exec time: {total_time:.3f}s")
    return jsonify({"detected_objects": results})
