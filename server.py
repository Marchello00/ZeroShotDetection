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

    queries = request.json.get("queries", {})
    image_folder = request.json.get("image_folder", ".")

    logger.info(f"got request: image_folder={image_folder}, queries={queries}")

    results = {}
    for img, labels in queries.items():
        try:
            img_path = f"{image_folder}/{img}"
            image = Image.open(img_path)
            results_local = search_on_image(image, labels)
            results[img] = [[region.label, *region.to_xywh()]
                            for region in results_local]
        except Exception as exc:
            logger.exception(exc)
            sentry_sdk.capture_exception(exc)
            results[img] = []

    total_time = time.time() - st_time
    logger.info(f"zero-shot object detection exec time: {total_time:.3f}s")
    return jsonify(results)
