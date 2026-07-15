# hash:sha256:4fb45452052ee461b51de861d5fd90b5bd6e73215af211d6dbffa60fe584097e
FROM registry.codeocean.com/codeocean/ubuntu:22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gdal-bin=latest \
        libgdal-dev=latest \
        python3=latest \
        python3-dev=latest \
        python3-pip=latest \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    rasterio==latest
