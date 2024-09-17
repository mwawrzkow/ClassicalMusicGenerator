# Use build argument for base image
ARG IMAGE_NAME
FROM ${IMAGE_NAME}

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
            numpy matplotlib jupyterlab music21 tqdm Tornado mido pretty_midi pandas


# # Set up user and group based on environment variables UID and GID
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g ${GID} devgroup && useradd -u ${UID} -g devgroup -m devuser && \
#     echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# # Set the user as the default user for the container
# USER devuser

# Set up the working directory
WORKDIR /workspace

# Create the midi_data directory with proper permissions
# RUN mkdir -p /workspace/midi_data && chown -R devuser:devgroup /workspace/midi_data

# Copy all files from src/ directory on host to /workspace inside the container
COPY ./src /workspace

# Expose necessary ports (e.g., for Flask)
EXPOSE 5000/tcp

# Set DISPLAY for GUI applications
ENV DISPLAY=${DISPLAY}
ENV QT_X11_NO_MITSHM=1
ENV IGNORE_NO_GPU=True

# Command to run when the container starts
CMD ["python", "/workspace/website.py"]
