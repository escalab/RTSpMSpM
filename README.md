# RTSpMSpM: Harnessing Ray Tracing for Efficient Sparse Matrix Computations

This repository contains the code and benchmark suite for **RTSpMSpM**, a novel approach that leverages NVIDIA’s hardware-accelerated ray tracing (RT Cores) to speed up **Sparse Matrix × Sparse Matrix Multiplication (SpMSpM)**. This project demonstrates the feasibility and benefits of mapping sparse matrix operations to the ray tracing pipeline.


## Technologies Used

- **Languages**: C++, Python
- **GPU Frameworks**: CUDA 12.3, NVIDIA OptiX 8.0.0, cuSPARSE
- **Build Tools**: CMake 3.22, GCC 7.5.0
- **Containers**: Docker 27.3.1 with NVIDIA support
- **Datasets**: SuiteSparse Matrix Collection

## Project Structure

```
RTSpMSpM/
├── cuSparse/                # GPU baseline using cuSPARSE
├── Dockerfile/              # Docker build scripts
├── optixSpMSpM/             # OptiX SDK and build system
│   ├── build/               # Compiled binaries and CMake output
│   └── src/            
│       ├── data/            # Input Datasets
│       ├── support/
│       ├── sutil/     
│       └── optixSpMSpM/     # Core ray tracing-based SpMSpM logic
└── Tool/
    └── Script/
        ├── AE_test.py               # Main script to launch experiments and benchmark
        ├── install.sh               # Compile program
        └── download_dataset.sh      # Benchmark automation script
```


## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/escalab/RTSpMSpM.git
cd RTSpMSpM
```

### Step 2: Build the Docker Image
```bash
cd Dockerfile
./build_image.sh
```

### Step 3: Start the Docker Container
```bash
./start_image.sh
```

### Step 4: Enter the Docker Container
```bash
./run.sh
```

### Step 5: Install and Compile Inside the Container
```bash
cd RTSpMSpM/scripts
./install.sh
```


## How to Run

If runned the installed script, skip to step 3: To run the experiment

### To compile the GPU Baseline:
```bash
cd /home/RTSpMSpM/cuSparse/src
make
```

### To compile the RT-based SpMSpM implementation:
```bash
mkdir -p /home/RTSpMSpM/optixSpMSpM/build
cd /home/RTSpMSpM/optixSpMSpM/build
cmake ../src
make
```

### To run the experiment:
```bash
python3 /home/RTSpMSpM/scripts/AE_test.py
```


## 6. Artifact Details

- **Artifact Availability**: Public  
  [📦 Zenodo Archive](https://zenodo.org/record/8210452)  
  [💻 GitHub Repo](https://github.com/escalab/RTSpMSpM)

- **Expected Output**:  
  Execution time (latency in milliseconds) for sparse matrix benchmarks, shown in logs or console output.

- **Evaluation Time**:
  - Setup: ~10 minutes
  - Experiment Runtime: 2–3 hours

- **Hardware Requirements**:
  - GPU: NVIDIA GPU with compute capability 5.0+ (7.5 recommended)
  - CPU: Original Hardware Intel Core i7 14700K
  - RAM: Original Hardware 128GB DDR4
  - Disk Space: up to ~2GB per datasets

- **Software Requirements**:
  - CUDA 12.3
  - Docker 27.3.1 (or cmake 3.22 + gcc 7.5.0 if building natively)
  - nvidia-docker recommended for easy setup

- **Licensing**:
  - Code: MIT License
  - Datasets: Original SuiteSparse licenses

---

## 7. Citation

If you use this artifact in your research, please cite the corresponding ISCA 2025 paper:

> *RT+SpMSpM: Harnessing Ray Tracing for Efficient Sparse Matrix Computations*, ISCA 2025.
