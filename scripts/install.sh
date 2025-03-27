DATA_DIR="/home/RTSpMSpM/optixSpMSpM/src/data"
OPTIX_DIR="/home/RTSpMSpM/optixSpMSpM"
CUSPARSE_DIR="/home/RTSpMSpM/cuSparse/src"

# Compile Optix SPMSPM
cd ${OPTIX_DIR}
mkdir build
cd build
cmake ../src
make
#TODO: Run OPTIX SPMSPM

# Compile cuSparse
cd ${CUSPARSE_DIR}
make
#TODO: Run Cusparse 