#!/bin/bash
export OPTIX_INSTALL_PATH=/home/OTEMP/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:$OPTIX_INSTALL_PATH/lib64"
