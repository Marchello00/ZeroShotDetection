# syntax=docker/dockerfile:experimental

FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

WORKDIR /src

ARG SERVICE_PORT
ENV SERVICE_PORT ${SERVICE_PORT}

# gcc for detectron2, ffmpeg/libsm6/libxext6 for opencv (https://stackoverflow.com/a/63377623)
RUN apt-get update && apt-get -y install gcc ffmpeg libsm6 libxext6

COPY ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY ./docker_cache_models.py /src/docker_cache_models.py
RUN python /src/docker_cache_models.py

COPY . /src

HEALTHCHECK --interval=5s --timeout=90s --retries=3 CMD curl --fail 127.0.0.1:${SERVICE_PORT}/healthcheck || exit 1


CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300