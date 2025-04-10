#!/bin/bash
DATA_DIR="/home/RTSpMSpM/optixSpMSpM/src/data"

# List of dataset URLs
urls=(
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/p2p-Gnutella31.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/mario002.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/web-Google.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/amazon0312.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-CondMat.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/email-Enron.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/wiki-Vote.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage12.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Um/2cubes_sphere.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Um/offshore.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Williams/cop20k_A.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/filter3D.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Da.tar.gz"
)

mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# Download and extract each file
for url in "${urls[@]}"; do
  echo "Processing $url"
  filename=$(basename "$url")
  dirname="${filename%.tar.gz}"

  # Download the tar.gz file
  if [ ! -f "$filename" ]; then
    wget "$url"
  else
    echo "$filename already exists. Skipping download."
  fi

  # Extract it if not already extracted
  if [ ! -d "$dirname" ]; then
    tar -xzf "$filename"
  else
    echo "$dirname already extracted. Skipping."
  fi

  rm -f "$filename"
done

echo "All datasets downloaded and extracted."