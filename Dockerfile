# base image you want to use
# make sure to use a CUDA image if running on GPUs
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04
FROM python:3.9.6-slim

# these two lines are mandatory
WORKDIR /gridai/project
COPY . .

RUN apt-get -y update
RUN apt-get -y install git
RUN pip install \
    pandas>="1.2.4" \
    matplotlib>="3.4.1" \
    numpy>="1.20.2" \
    pint>="0.17" \
    python-dotenv>="0.17.0" \
    openpyxl>="3.0.7" \
    ord-schema>="0.3.0" \
    rdkit-pypi>="2021.3.2" \
    typer>="0.3.2" \
    tqdm>="4.61.0" \
    git+git://github.com/marcosfelt/typer@master#egg=typer \
    git+git://github.com/sustainable-processes/summit@master#egg=summit \
    absl-py>=0.9.0 \
    flask>=1.1.2 \
    protobuf>=3.13.0 \
    protoc-wheel-0>=3.14.0 \
    pygithub>=1.51 \
    python-dateutil>=1.10.0 \
    jinja2>=2.0.0 \
    xlrd<=2.0.0 \ 
    xlwt>=1.3.0 \
    joblib>=1.0.0