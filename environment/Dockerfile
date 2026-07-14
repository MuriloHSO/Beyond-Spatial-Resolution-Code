# hash:sha256:fec95480dfb261dd4a20bde6597748f56b6adf499d6a7521b424af23a2f93dd4
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        gcc==latest \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    gcc==latest
