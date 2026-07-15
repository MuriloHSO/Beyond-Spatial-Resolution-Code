# hash:sha256:ab000e01782d338df8aeeb66a792c631b484f095cdcbd98b74001d9b0f903513
FROM registry.codeocean.com/codeocean/ubuntu:22.04.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip=latest \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/*
