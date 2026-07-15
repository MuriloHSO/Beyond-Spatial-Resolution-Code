# hash:sha256:a10ad140de368b827b7766357a12cfd2c0d3caceb4823601500306d7e3bfe214
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
    matplotlib==latest \
    numpy==latest \
    openpyxl==latest \
    pandas==latest \
    rasterio==latest \
    scikit-learn
