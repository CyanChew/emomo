# Use an official Ubuntu as a parent image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Google Chrome
RUN apt-get update && apt-get install -y wget gnupg && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
    apt-get update && apt-get install -y google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

# Update apt and install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    vim \
    curl \
    gnupg2 \
    sudo \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    && apt-get clean

# Install Miniconda (which includes mamba)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init \
    && /opt/conda/bin/conda install -y mamba -c conda-forge

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy the requirements.txt file into the container
COPY requirements.txt /tmp/requirements.txt

# Create the conda environment and install dependencies using mamba
RUN mamba create -n tidybot2 python=3.10.14 && \
    echo "conda activate tidybot2" >> ~/.bashrc

# Set the working directory to /tmp
WORKDIR /tmp

# Download the kortex_api and opencv_python wheels
RUN curl -L "https://www.dropbox.com/scl/fi/behem9thid9benovogfdf/kortex_api-2.6.0.post3-py3-none-any.whl?rlkey=13zikoe4cwtk59q86jfx3a2k5&dl=1" -o /tmp/kortex_api-2.6.0.post3-py3-none-any.whl && \
    curl -L "https://www.dropbox.com/scl/fi/9gr7u1dw3ef5x73ppkpxi/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl?rlkey=8bh4tl7is0ixp69zm9ohko0yb&st=5prf2cxo" -o /tmp/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl

RUN apt update && apt install -y build-essential manpages-dev wget

# Download the kortex_api and opencv_python wheels
RUN curl -L "https://www.dropbox.com/scl/fi/behem9thid9benovogfdf/kortex_api-2.6.0.post3-py3-none-any.whl?rlkey=13zikoe4cwtk59q86jfx3a2k5&dl=1" -o /tmp/kortex_api-2.6.0.post3-py3-none-any.whl && \
    curl -L "https://www.dropbox.com/scl/fi/9gr7u1dw3ef5x73ppkpxi/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl?rlkey=8bh4tl7is0ixp69zm9ohko0yb&st=5prf2cxo" -o /tmp/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl

# Install kortex_api and opencv_python dependencies
RUN /opt/conda/bin/conda run -n tidybot2 pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda run -n tidybot2 pip install /tmp/kortex_api-2.6.0.post3-py3-none-any.whl

RUN /opt/conda/bin/conda run -n tidybot2 pip install protobuf==3.20.0 && \
    /opt/conda/bin/conda run -n tidybot2 pip install --force-reinstall --no-deps /tmp/opencv_python-4.9.0.80-cp310-cp310-linux_x86_64.whl

RUN apt install net-tools
RUN apt install iputils-ping
RUN apt-get update && apt-get install -y usbutils

WORKDIR /host
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
RUN /opt/conda/bin/conda run -n tidybot2 python --version
# Install TensorFlow, Selenium, and Accelerate in the tidybot2 env
RUN /opt/conda/bin/conda run -n tidybot2 pip install selenium einops
# Install Chrome
RUN apt-get update && apt-get install -y wget gnupg2
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get update && apt-get install -y google-chrome-stable
