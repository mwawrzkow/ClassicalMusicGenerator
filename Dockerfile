# Use NVIDIA's TensorFlow image as the base
FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3

# Set environment variables to use the NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Set the shared memory size
RUN echo "kernel.shmmax = 34359738368" >> /etc/sysctl.conf

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sudo \
    htop \
    git \
    wget \
    vim \
    curl \
    build-essential \
    pkg-config \
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install flask Flask-SocketIO eventlet gevent flask-cors \
                numpy matplotlib jupyterlab

# Set up user and group based on environment variables UID and GID
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} devgroup && useradd -u ${UID} -g devgroup -m devuser && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the user as the default user for the container
USER devuser

# Set up the working directory
WORKDIR /workspace

# Copy all files from src/ directory on host to /workspace inside the container
COPY ./src /workspace

# Expose any necessary ports (e.g., for Flask or JupyterLab)
EXPOSE 5000

# Set DISPLAY for GUI applications
ENV DISPLAY=${DISPLAY}
ENV QT_X11_NO_MITSHM=1

# Command to run when the container starts
CMD ["python", "/workspace/website.py"]
