# Intro to Optix Project

# 1. Project Overview
    - TODO
    - **Brief Description**: Provide a high-level description of the project. Mention the main goals and the problems it solves.
    - **Features**: Highlight the main features or functionalities of the project.


# 2. Technologies Used

The project leverages a variety of cutting-edge technologies, frameworks, and libraries to achieve efficient GPU computation, data analysis, and high-performance rendering. Here is an overview of the main technologies used:

- C++
- nvcc
- cuda
- optix 8.0.0
- intel oneAPI
- cuSparse
- PaRMAT
- python
- docker
# 3. Project Structure
    OptixSDK
    ├── cuSparse/                 # GPU Baseline
    ├── Dockerfile/      
    ├── NVIDIA-OptiX-SDK-8.0.0/   # 
    |   ├── build                 # binary and CMakeFiles
    |   └── SDK                   
    |        └── optixSphere      # src code
    └── Tool/                     # Helper scripts
        └── Script/
             ├── src/gamma_test.py             # Top-most script to run everything
             └── src/mmNsys_increased_size.sh  # Simpler script with commands


# 4. Installation and Setup

To set up the environment and install all dependencies, use the Docker container. The Docker configuration ensures a smooth and consistent setup process.


## **Step 1. Clone the Repository**
    git clone <repository-url> 
    cd OptixSDK


## **Step 2: Build the Docker Image** 

Navigate to the `Dockerfile/` directory and use the provided `build.sh` script to build the Docker image.

    cd Dockerfile/
    ./build.sh

This script handles building the Docker image with all required dependencies (OptiX SDK, CUDA, etc.).

## **Step 3: Start the Docker Container** Use the `start_image.sh` script to start the container.
    ./start_image.sh


## **Step 4: Access the Docker Container** To enter the running container, use the `run.sh` script:
    ./run.sh


# 5. How to Run

The project should be already built. All you need to do is to compile the binaries and everything should be runnable.


## **How to compile GPU baseline**

**Baseline Structure**

    OptixSDK
    └── cuSparse/  
          └── src/

**Compile**
bash

    cd OptixSDK/cuSparse/src
    make


## **How to compile Solution**

**Project Structure**

    OptixSDK
    └── NVIDIA-OptiX-SDK-8.0.0/  
          ├── build/
          └── src/
               ├── sutil/
               └── optixSphere/

**Compile Project**
bash

    mkdir build
    cd OptixSDK/NVIDIA-OptiX-SDK-8.0.0/build
    cmake ../SDK
    make
    
**Compile**
bash
    cd OptixSDK/NVIDIA-OptiX-SDK-8.0.0/build
    make
This will compile the project into runnable as `OptixSDK/NVIDIA-OptiX-SDK-8.0.0/build/bin/optixSphere`


## How to run the Project
You can execute the project with gamma dataset with the script `OptixSDK/Tool/Script/src/gamma_test.py`

Ulternatively, you can execute the main script `mmNsys_increased_size.sh` to start the entire workflow:
bash

    cd OptixSDK/Tool/Script/src
    ./mmNsys_increased_size.sh

This script orchestrates the different components, including GPU benchmarks and performance profiling using NVIDIA Nsight Systems (`nsys`).


# 7. Troubleshooting and Common Issues
## Common Problem
- For some reason docker container sometimes doesn’t install some python packages upon building the container, please use `pip3 install` when needed.

