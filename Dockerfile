FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install Python 3.13
RUN apt-get update && apt-get install -y wget software-properties-common build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.13 python3.13-venv python3.13-dev ninja-build && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Sharp and dependencies
RUN mkdir /app
COPY pyproject.toml requirements.txt requirements.in /app/
COPY src/ /app/src/
WORKDIR /app
RUN python3.13 -m venv .venv
ENV TORCH_CUDA_ARCH_LIST="6.1;8.0"
ENV FORCE_CUDA="1"
RUN .venv/bin/pip install ninja
RUN .venv/bin/pip install -r requirements.txt
RUN .venv/bin/pip install gradio
RUN ln -s /app/.venv/bin/sharp /usr/local/bin/sharp

# Test run to download model and check if it works
RUN wget https://apple.github.io/ml-sharp/thumbnails/Unsplash_-5wkyNA2BPc_0000-0001.jpg -O /tmp/test.jpg
RUN sharp predict -i /tmp/test.jpg -o /tmp/test
RUN rm /tmp/test.jpg /tmp/test -rf

RUN .venv/bin/pip uninstall -y torch
RUN .venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN .venv/bin/pip install -y flask
RUN .venv/bin/python3.13 -c "import torch;torch.hub.load_state_dict_from_url('https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt')"

# Copy other files
COPY . /app

# Start Gradio web server
CMD [".venv/bin/python3.13", "-u", "/app/webui.py", "--preload"]
