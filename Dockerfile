FROM docker.io/pytorch/pytorch:latest

WORKDIR /VSTE

# Install git and make repository available
RUN apt-get update && apt-get install -y git
RUN git config --global --add safe.directory /VSTE

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt
