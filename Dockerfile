# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y build-essential git
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
RUN apt-get install -y ffmpeg

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

CMD uvicorn api:app