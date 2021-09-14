import logging
import time

from PIL import Image
from flask import Flask, request, jsonify
from healthcheck import HealthCheck
import sentry_sdk

from pipeline import search_on_image

logger = logging.getLogger(__name__)

app = Flask(__name__)
health = HealthCheck(app, "/healthcheck")
logging.getLogger("werkzeug").setLevel("WARNING")


@app.route("/respond", methods=["POST"])
def respond():
    st_time = time.time()

    logger.info(f"got request: {request.json}")

    results = {}
    for img_path, labels in request.json.items():
        try:
            image = Image.open(img_path)
            results_local = search_on_image(image, labels)
            results[img_path] = [[region.label, *region.to_xywh()]
                                 for region in results_local]
        except Exception as exc:
            logger.exception(exc)
            sentry_sdk.capture_exception(exc)
            results[img_path] = []

    total_time = time.time() - st_time
    logger.info(f"zero-shot object detection exec time: {total_time:.3f}s")
    return jsonify(results)
