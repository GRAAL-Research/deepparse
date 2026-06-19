# syntax = docker/dockerfile:experimental
# Base images are pinned by digest (in addition to the tag) so the build is reproducible and not silently
# swapped under a mutable tag.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip
RUN CFLAGS="-g0 -Os -DNDEBUG -Wl,--strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib" \
    DEEPPARSE_RELEASE_BUILD=1 \
    pip3 install --no-cache-dir \
    --compile \
    --global-option=build_ext \
    --global-option="-j 4" \
    --no-deps -U git+https://github.com/GRAAL-Research/deepparse.git@stable

RUN find /opt/conda/lib/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.txt' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.mc' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/ -name '*.c' -delete \
    && find /opt/conda/lib/ -name '*.pxd' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.md' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.png' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.jpg' -delete \
    && find /opt/conda/lib/ -follow -type f -name '*.jpeg' -delete \
    && find /opt/conda/lib/ -name '*.pyd' -delete \
    && find /opt/conda/lib/ -name '__pycache__' | xargs rm -r

ENV PATH /opt/conda/bin:$PATH


FROM python:3.13-slim@sha256:c33f0bc4364a6881bed1ec0cc2665e6c53c87a43e774aaeab88e6f17af105e4f AS app

# set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=local

# set work directory
WORKDIR /app

# copy project
COPY /deepparse ./deepparse
COPY setup.cfg ./
COPY setup.py ./
COPY README.md ./
COPY version.txt ./
RUN pip install -e .[app]

# Run as a non-root user (least privilege). The model cache lives under this user's home.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser