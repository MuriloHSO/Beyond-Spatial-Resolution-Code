# hash:sha256:2ba5f41b0512e551e2073ab9daf47fdef8155cf3c56a713d25c1ef3c05b96b19
FROM registry.codeocean.com/codeocean/ubuntu:22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=latest \
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
