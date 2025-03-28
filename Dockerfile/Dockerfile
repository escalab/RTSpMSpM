# Use NVIDIA CUDA base image
ARG BASE_IMG=nvidia/cuda:12.3.0-devel-ubuntu22.04
FROM ${BASE_IMG}

# Set timezone (modify as needed)
ARG CONTAINER_TIMEZONE=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && \
    echo $CONTAINER_TIMEZONE > /etc/timezone

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y vim cmake g++ unzip git wget python3-pip && \
    pip3 install cmake --upgrade

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh && \
    apt-get install -y gnuplot && \
    pip3 install numpy scipy

# Download and install Nsight Systems
# Install required dependencies
RUN apt-get update && \
    apt-get install -y wget libxcb-cursor0 libnss3 libasound2 libopengl0 && \
    rm -rf /var/lib/apt/lists/*
# RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb && \
#     dpkg -i nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb && \
#     rm nsight-systems-2024.5.1_2024.5.1.113-1_amd64.deb

# Set environment variables for Miniconda and binaries
ENV PATH /root/miniconda3/bin:$PATH
ENV PATH /usr/local/bin:$PATH

# Install Optix Libraries
# OpenGL, RandR, Xinerama, Xcursor
RUN apt-get update && \
    apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev && \
    apt-get install -y libxrandr-dev libxinerama-dev libxcursor-dev && \
    apt-get install -y x11-apps
# Install cuSparse
RUN apt-get update && \
    apt-get install --only-upgrade libstdc++6

# # For intelMKL
# RUN apt-get install gpg-agent && \
#     wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg && \
#     echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list
# RUN apt-get update && \
#     apt-get install -y intel-basekit


# Set environment variables
ENV PATH "/usr/bin:${PATH}"
# ENV PATH=$PATH:/opt/nvidia/nsight-compute/2023.3.0/host/target-linux-x64
ENV LD_LIBRARY_PATH "/home/lib:${LD_LIBRARY_PATH}"

# # Install NVIDIA OptiX SDK
# RUN wget https://developer.nvidia.com/downloads/designworks/optix/secure/8.0.0/nvidia-optix-sdk-8.0.0-linux64-x86_64.sh
# RUN chmod +x NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh && \
#     ./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --skip-license --prefix=/opt/NVIDIA-OptiX-SDK-8.0.0

# # Set environment variables for OptiX SDK
# ENV OPTIX_SDK_PATH=/opt/NVIDIA-OptiX-SDK-8.0.0
# ENV LD_LIBRARY_PATH=${OPTIX_SDK_PATH}/lib64:${LD_LIBRARY_PATH}
# ENV OPTIX_ROOT=/opt/optix/NVIDIA-OptiX-SDK-7.0.0-linux64
# ENV PATH=$PATH:$OPTIX_ROOT/bin
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTIX_ROOT/lib64
ENV OPTIX_INSTALL_PATH /home/RTSpMSpM/optixSpMSpM
ENV -v /tmp/.X11-unix:/tmp/.X11-unix
ENV NVIDIA_DRIVER_CAPABILITIES all
WORKDIR /home

# Define entry point
# CMD ["bash"]
