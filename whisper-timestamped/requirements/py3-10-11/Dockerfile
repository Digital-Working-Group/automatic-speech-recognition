FROM continuumio/miniconda3
RUN conda create -y -n env python=3.10.12
RUN echo "source activate env" > ~/.bashrc
RUN conda install -c conda-forge ffmpeg -y
ENV PATH /opt/conda/envs/env/bin:$PATH
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip
WORKDIR /scripts