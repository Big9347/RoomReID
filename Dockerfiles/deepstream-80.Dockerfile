# set base image (host OS)
FROM nvcr.io/nvidia/deepstream:8.0-gc-triton-devel

# set the working directory in the container
WORKDIR /opt/nvidia/deepstream/deepstream-8.0/sources/apps/sample_apps/deepstream-fewshot-learning-app

# copy the dependencies file to the working directory
COPY ./deepstream/deepstream-fewshot-learning-app/ ./deepstream/init_scripts/*  ./
COPY ./deepstream/models/* ./models/
COPY ./deepstream/videos/* ./videos/