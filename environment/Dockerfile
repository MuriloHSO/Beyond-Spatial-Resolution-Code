# hash:sha256:72e104d2fc34f198acd229894c11ad18adf8f853acad4facde828e22792e4e16
FROM registry.codeocean.com/codeocean/ubuntu:22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        gdal-bin=3.4.1+dfsg-1build4 \
        libgdal-dev=3.4.1+dfsg-1build4 \
        python3=3.10.6-1~22.04.1 \
        python3-dev=3.10.6-1~22.04.1 \
        python3-pip=22.0.2+dfsg-1ubuntu0.7 \
        python3-setuptools=59.6.0-1.2ubuntu0.22.04.3 \
        python3-wheel=0.37.1-2ubuntu0.22.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    huggingface-hub==1.24.0 \
    matplotlib==3.10.9 \
    numpy==2.2.6 \
    openpyxl==3.1.5 \
    pandas==2.3.3 \
    rasterio==1.4.4 \
    scikit-learn==1.7.2 \
    tomli==2.4.1
