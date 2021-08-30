# syntax=docker/dockerfile:experimental

FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /src

ARG SERVICE_PORT
ENV SERVICE_PORT ${SERVICE_PORT}

RUN apt-get update && apt-get -y install gcc

COPY ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

RUN python -c 'from torchvision.models import inception_v3; inception_v3(pretrained=True);'

COPY . /src

HEALTHCHECK --interval=5s --timeout=90s --retries=3 CMD curl --fail 127.0.0.1:${SERVICE_PORT}/healthcheck || exit 1


CMD gunicorn --workers=1 server:app -b 0.0.0.0:${SERVICE_PORT} --timeout=300