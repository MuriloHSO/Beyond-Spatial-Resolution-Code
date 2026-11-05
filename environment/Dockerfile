FROM registry.codeocean.com/codeocean/ubuntu:22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN pip3 install --no-cache-dir uv==0.13.7 \
    && uv venv /opt/venv --python 3.14.7 \
    && uv pip install --python /opt/venv/bin/python --no-cache \
        huggingface_hub==1.26.0 \
        joblib==1.5.3 \
        matplotlib==3.11.1 \
        numpy==2.5.1 \
        pandas==3.0.5 \
        pyogrio==0.13.0 \
        rasterio==1.5.0 \
        scikit-learn==1.9.0 \
        scipy==1.18.0 \
        shapely==2.2.0 \
        threadpoolctl==3.7.0

ENV PATH="/opt/venv/bin:$PATH"
