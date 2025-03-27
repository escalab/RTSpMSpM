# /home/RTSpMSpM/scripts/ae_run.sh

DATA_DIR="/home/RTSpMSpM/optixSpMSpM/src/data"
OPTIX_BUILD="/home/RTSpMSpM/optixSpMSpM/build"
CUSPARSE_DIR="/home/RTSpMSpM/cuSparse/src"
OPTIX_DIR="/home/RTSpMSpM/optixSpMSpM"

# # Run Optix
# cd ${OPTIX_BUILD}
# ${OPTIX_BUILD}/bin/optixSpMSpM -m1 "./Matrix/test1.mtx" -m2 "./Matrix/test2.mtx" -o "./test_output.mtx" -l "/home/RTSpMSpM/scripts/temp.txt"

# # Run Cusparse 
# cd ${CUSPARSE_DIR}
# ${CUSPARSE_DIR}/cuSparse -m1 "${DATA_DIR}/Matrix/test1.mtx" -m2 "${DATA_DIR}/Matrix/test2.mtx" -o "${DATA_DIR}/test_output.mtx" -l "/home/RTSpMSpM/scripts/temp2.txt"

python3 AE_test.py