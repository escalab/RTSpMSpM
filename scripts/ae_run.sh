# /home/RTSpMSpM/scripts/ae_run.sh

DATA_DIR="/home/RTSpMSpM/optixSpMSpM/src/data"
OPTIX_BUILD="/home/RTSpMSpM/optixSpMSpM/build"
CUSPARSE_DIR="/home/RTSpMSpM/cuSparse/src"
OPTIX_DIR="/home/RTSpMSpM/optixSpMSpM"

# Compile Optix SPMSPM
cd ${OPTIX_BUILD}
make
# Run Optix
${OPTIX_BUILD}/bin/optixSpMSpM -m1 "./Matrix/test1.mtx" -m2 "./Matrix/test2.mtx" -o "./test_output.mtx"

# Compile cuSparse
cd ${CUSPARSE_DIR}
make
${CUSPARSE_DIR}/cuSparse -m1 "${DATA_DIR}/Matrix/test1.mtx" -m2 "${DATA_DIR}/Matrix/test2.mtx" -o "${DATA_DIR}/test_output.mtx"
#TODO: Run Cusparse 
