FROM python:3.10.12-slim-bullseye

RUN apt-get update && apt-get install -y ffmpeg

RUN --mount=type=cache,target=/root/.cache/pip --mount=source=requirements.txt,target=requirements.txt \
    python3 -m pip install -r requirements.txt

WORKDIR /scripts/src
COPY src/. .

# FROM continuumio/miniconda3
# RUN conda create -y -n env python=3.10.12
# RUN echo "source activate env" > ~/.bashrc
# RUN conda install conda-forge ffmpeg
# ENV PATH /opt/conda/envs/env/bin:$PATH

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip

# WORKDIR /src
# COPY src/. .
