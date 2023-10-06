FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0
#FROM pytorch/pytorch
USER root
ENV PYTHONPATH "${PYTHONPATH}:/src"
WORKDIR /
ENTRYPOINT ["python3"]
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt pip install --upgrade torch torchvision
RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y gcc
RUN --mount=type=cache,target=/var/cache/apt pip install pandas
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y openslide-tools
RUN --mount=type=cache,target=/var/cache/apt pip install openslide-python
RUN --mount=type=cache,target=/var/cache/apt pip install histolab
RUN --mount=type=cache,target=/var/cache/apt pip install plotly
RUN --mount=type=cache,target=/var/cache/apt pip install torchmetrics
RUN --mount=type=cache,target=/var/cache/apt pip install wandb
RUN --mount=type=cache,target=/var/cache/apt pip install py-params
RUN --mount=type=cache,target=/var/cache/apt pip install torchstain
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y git
RUN --mount=type=cache,target=/var/cache/apt pip install git+https://github.com/lucasrla/wsi-tile-cleanup.git
RUN --mount=type=cache,target=/var/cache/apt pip install pyvips
RUN apt-get -yq update
RUN --mount=type=cache,target=/var/cache/apt apt-get -yq install libvips
RUN --mount=type=cache,target=/var/cache/apt pip install tdqm