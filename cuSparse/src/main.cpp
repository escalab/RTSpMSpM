/**
 * cuSparse spmspm
 * 
 * Purpose: Baseline for spmspm on gpu side.
 *
 * To Compile:
 *    Run command "make" in the build folder
 * To run: 
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen1024_1048.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen1024_1048.mtx" -o "./out.mtx"
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen8192_67108.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen8192_67108.mtx" -o "./out.mtx"
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen4096_16777.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen4096_16777.mtx" -o "./out.mtx"
 *
 * Nsys command profiling:
 * nsys nvprof ./cuSparse -m1 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" -m2 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx”
 * 
 * To Debug:
 * cuda-gdb /home/OptixSDK/cuSparse/src/cuSparse 
 * run -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen16384_268435.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen16384_268435.mtx" -o "./out.mtx"
*/

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpSM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono> 
using namespace std::chrono;

#include "util.cpp"
#include "Timing.cpp"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --output | -o <filename>    Specify file for image output\n";
    std::cerr << "         --mat1 | -m1 <filename>     Specify file for matrix 1\n";
    std::cerr << "         --mat2 | -m2 <filename>     Specify file for matrix 2\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 1 );
}

void getComputeType( cusparseSpMatDescr_t mat )
{
    cudaDataType dataType;
    size_t dataSize = sizeof(cudaDataType);

    // Query the data type of the sparse matrix
    cusparseSpMatGetAttribute(
        mat, 
        CUSPARSE_SPMAT_DIAG_TYPE, 
        &dataType, 
        dataSize
    );

    switch (dataType) {
        case CUDA_R_32F:
            std::cout << "Matrix data type: CUDA_R_32F (float)" << std::endl;
            break;
        case CUDA_R_64F:
            std::cout << "Matrix data type: CUDA_R_64F (double)" << std::endl;
            break;
        case CUDA_C_32F:
            std::cout << "Matrix data type: CUDA_C_32F (complex float)" << std::endl;
            break;
        case CUDA_C_64F:
            std::cout << "Matrix data type: CUDA_C_64F (complex double)" << std::endl;
            break;
        default:
            std::cout << "Unknown data type" << std::endl;
            break;
    }
}

int reuseCompute( std::string matrix1File, std::string matrix2File, std:: string outfile )
{
    // Start timer
    Timing::reset();

    // Host Matrix Definition 
    int* hA_mat_rows, *hB_mat_rows, *hC_mat_rows;
    int* hA_mat_cols, *hB_mat_cols, *hC_mat_cols;
    float* hA_mat_values, *hB_mat_values, *hC_mat_svalues;
    int hA_mat_num_rows, hB_mat_num_rows, hC_mat_num_rows;
    int hA_mat_num_cols, hB_mat_num_cols, hC_mat_num_cols;
    uint64_t hA_mat_nnz, hB_mat_nnz, hC_mat_nnz;

    cooFromFile(matrix1File, &hA_mat_rows, &hA_mat_cols, &hA_mat_values, &hA_mat_num_rows,
                &hA_mat_num_cols, &hA_mat_nnz);
    cooFromFile(matrix2File, &hB_mat_rows, &hB_mat_cols, &hB_mat_values, &hB_mat_num_rows,
                &hB_mat_num_cols, &hB_mat_nnz);

    Timing::startTiming("computation time no io");

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device Memory Management
    int   *dA_rows, *dA_cols, *dB_rows, *dB_cols, *dC_rows, *dC_cols;
    float *dA_values, *dB_values, *dC_values;
    int   *dA_csrPtr, *dB_csrPtr, *dC_csrPtr;
    // Change coo to csr Uncomment if >64
    // hA_csr_rows = new int[hA_mat_num_rows + 1];
    // hB_csr_rows = new int[hB_mat_num_rows + 1];
    // coo_to_csr(hA_mat_rows, hA_mat_nnz, hA_mat_num_rows, hA_csr_rows);
    // coo_to_csr(hB_mat_rows, hB_mat_nnz, hB_mat_num_rows, hB_csr_rows);

    // Allocate device memory for matrix A
    CHECK_CUDA( cudaMalloc((void**)&dA_cols, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dA_values, hA_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dA_csrPtr, (hA_mat_num_rows + 1) * sizeof(int)));

    // Allocate device memory for matrix B
    CHECK_CUDA( cudaMalloc((void**)&dB_cols, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_values, hB_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dB_csrPtr, (hB_mat_num_rows + 1) * sizeof(int)));

    CHECK_CUDA( cudaMalloc((void**) &dC_csrPtr,
                        (hA_mat_num_rows + 1) * sizeof(int)) );

    // Copy matrix A data from host to device
    CHECK_CUDA( cudaMemcpy(dA_cols, hA_mat_cols, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dA_values, hA_mat_values, hA_mat_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Copy matrix B data from host to device
    CHECK_CUDA( cudaMemcpy(dB_cols, hB_mat_cols, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_values, hB_mat_values, hB_mat_nnz * sizeof(float), 
                        cudaMemcpyHostToDevice));
    
    // Comment out if >64
    CHECK_CUDA( cudaMalloc((void**)&dA_rows, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_rows, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMemcpy(dA_rows, hA_mat_rows, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_rows, hB_mat_rows, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL;
    void*  dBuffer2    = NULL;
    void*  dBuffer3    = NULL;
    void*  dBuffer4    = NULL;
    void*  dBuffer5    = NULL;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    // Change matrix row ptr to CSR format (Uncomment if >64)
    // CHECK_CUDA( cudaMemcpy(dA_csrPtr, hA_csr_rows, (hA_mat_num_rows + 1) * sizeof(int),
    //                     cudaMemcpyHostToDevice));
    // CHECK_CUDA( cudaMemcpy(dB_csrPtr, hB_csr_rows, (hB_mat_num_rows + 1) * sizeof(int),
    //                     cudaMemcpyHostToDevice));

    // Comment out if >64
    cusparseXcoo2csr(handle, dA_rows, hA_mat_nnz, hA_mat_num_rows, dA_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );
    cusparseXcoo2csr(handle, dB_rows, hB_mat_nnz, hB_mat_num_rows, dB_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );

    // Create sparse matrix in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, hA_mat_num_rows, hA_mat_num_cols, hA_mat_nnz,
                                    dA_csrPtr, dA_cols, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, hB_mat_num_rows, hB_mat_num_cols, hB_mat_nnz,
                                    dB_csrPtr, dB_cols, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, hA_mat_num_rows, hB_mat_num_cols, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE,
                                    CUDA_R_32F) );

    //==========================================================================
    // SpGEMM Computation
    //==========================================================================
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                        CUSPARSE_SPGEMM_ALG2,
                                        spgemmDesc, &bufferSize1, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                        CUSPARSE_SPGEMM_ALG2,
                                        spgemmDesc, &bufferSize1, dBuffer1)
    )
    //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize2, NULL, &bufferSize3, NULL,
                                &bufferSize4, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer3, bufferSize3) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer4, bufferSize4) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                                &bufferSize4, dBuffer4)
    )
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    //--------------------------------------------------------------------------

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows, C_num_cols, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols,
                                        &C_nnz) )
    // allocate matrix C
    hC_mat_num_rows = (int)C_num_rows;
    hC_mat_num_cols = (int)C_num_cols;
    hC_mat_nnz = (uint64_t)C_nnz;

    CHECK_CUDA( cudaMalloc((void**) &dC_rows, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_cols, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )
    CHECK_CUDA( cudaMemset(dC_values, 0x0, C_nnz * sizeof(float)) )
    // fill dC_values if needed
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrPtr, dC_cols, dC_values) )
    //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize5, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer5, bufferSize5) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize5, dBuffer5)
    )
    CHECK_CUDA( cudaFree(dBuffer3) )
    //--------------------------------------------------------------------------
    // Compute
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                    matC, computeType, CUSPARSE_SPGEMM_ALG2,
                                    spgemmDesc)
    )
    // update dA_values, dB_values
    CHECK_CUDA( cudaMemcpy(dA_values, hA_mat_values,
                        hA_mat_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_mat_values,
                        hB_mat_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    // second run
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                    matC, computeType, CUSPARSE_SPGEMM_ALG2,
                                    spgemmDesc)
    )
    
    // Change csr to coo
    CHECK_CUSPARSE(
        cusparseXcsr2coo(handle, dC_csrPtr, hC_mat_nnz, hC_mat_num_rows, dC_rows,
                    CUSPARSE_INDEX_BASE_ONE)  )
    
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    cudaDeviceSynchronize();
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    // Finish Computation
    Timing::stopTiming(true);
    

    //--------------------------------------------------------------------------
    // device result load
    // int   hC_rows_tmp[hC_mat_nnz];
    int* hC_rows_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    // int   hC_columns_tmp[hC_mat_nnz];
    int* hC_columns_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    float* hC_values_tmp = new float[hC_mat_nnz];
    Timing::stopTiming(true);
    // Comment out if >64
    CHECK_CUDA( cudaMemcpy(hC_rows_tmp, dC_rows, hC_mat_nnz * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    // Uncomment if >64
    // int   hC_csr_row[(hC_mat_num_rows + 1)];
    // CHECK_CUDA( cudaMemcpy(hC_csr_row, dC_csrPtr, (hC_mat_num_rows + 1) * sizeof(int),
    //                     cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_cols, hC_mat_nnz * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, hC_mat_nnz * sizeof(float),
                        cudaMemcpyDeviceToHost) )

    // Uncomment if >64
    // csr_to_coo(hC_csr_row, hC_mat_num_rows, hC_mat_nnz, hC_rows_tmp, true);
    
    printCooToFile(outfile, hC_rows_tmp, hC_columns_tmp, hC_values_tmp, hC_mat_num_rows, hC_mat_num_cols, hC_mat_nnz);

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer4) )
    CHECK_CUDA( cudaFree(dBuffer5) )
    CHECK_CUDA( cudaFree(dA_csrPtr) )
    CHECK_CUDA( cudaFree(dA_cols) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrPtr) )
    CHECK_CUDA( cudaFree(dB_cols) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrPtr) )
    CHECK_CUDA( cudaFree(dC_cols) )
    CHECK_CUDA( cudaFree(dC_values) )
    free(hC_rows_tmp);
    free(hC_columns_tmp);
    free(hC_values_tmp);

    // Stop timer
    Timing::flushTimer();
    // auto stop = high_resolution_clock::now();
    // auto end2end = duration_cast<nanoseconds>(stop - start);
    // std::cout << "END_TO_END_LATENCY = " << end2end.count() << std::endl;

    return EXIT_SUCCESS;
}

int compute ( std::string matrix1File, std::string matrix2File, std:: string outfile ) {
    // Start timer
    auto start = high_resolution_clock::now();
    Timing::reset();

    // Host Matrix Definition 
    int* hA_mat_rows, *hB_mat_rows, *hC_mat_rows;
    int* hA_mat_cols, *hB_mat_cols, *hC_mat_cols;
    float* hA_mat_values, *hB_mat_values, *hC_mat_svalues;
    int hA_mat_num_rows, hB_mat_num_rows, hC_mat_num_rows;
    int hA_mat_num_cols, hB_mat_num_cols, hC_mat_num_cols;
    uint64_t hA_mat_nnz, hB_mat_nnz, hC_mat_nnz;

    cooFromFile(matrix1File, &hA_mat_rows, &hA_mat_cols, &hA_mat_values, &hA_mat_num_rows,
                &hA_mat_num_cols, &hA_mat_nnz);
    cooFromFile(matrix2File, &hB_mat_rows, &hB_mat_cols, &hB_mat_values, &hB_mat_num_rows,
                &hB_mat_num_cols, &hB_mat_nnz);

    Timing::startTiming("computation time no io");

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device Memory Management
    int   *dA_rows, *dA_cols, *dB_rows, *dB_cols, *dC_rows, *dC_cols;
    float *dA_values, *dB_values, *dC_values;
    int   *dA_csrPtr, *dB_csrPtr, *dC_csrPtr;

    // Allocate device memory for matrix A
    CHECK_CUDA( cudaMalloc((void**)&dA_cols, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dA_values, hA_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dA_csrPtr, (hA_mat_num_rows + 1) * sizeof(int)));

    // Allocate device memory for matrix B
    CHECK_CUDA( cudaMalloc((void**)&dB_cols, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_values, hB_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dB_csrPtr, (hB_mat_num_rows + 1) * sizeof(int)));

    CHECK_CUDA( cudaMalloc((void**) &dC_csrPtr,
                        (hA_mat_num_rows + 1) * sizeof(int)) );

    // Copy matrix A data from host to device
    CHECK_CUDA( cudaMemcpy(dA_cols, hA_mat_cols, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dA_values, hA_mat_values, hA_mat_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Copy matrix B data from host to device
    CHECK_CUDA( cudaMemcpy(dB_cols, hB_mat_cols, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_values, hB_mat_values, hB_mat_nnz * sizeof(float), 
                        cudaMemcpyHostToDevice));
    
    // Comment out if >64
    CHECK_CUDA( cudaMalloc((void**)&dA_rows, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_rows, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMemcpy(dA_rows, hA_mat_rows, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_rows, hB_mat_rows, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseSpGEMMAlg_t  alg    = CUSPARSE_SPGEMM_ALG3;
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    int64_t              num_prods;
    float                chunk_fraction = 0.01;
    void*  dBuffer1    = NULL, *dBuffer2   = NULL, *dBuffer3   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0,    bufferSize3 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) );

    cusparseXcoo2csr(handle, dA_rows, hA_mat_nnz, hA_mat_num_rows, dA_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );
    cusparseXcoo2csr(handle, dB_rows, hB_mat_nnz, hB_mat_num_rows, dB_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );

    // Create sparse matrix in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, hA_mat_num_rows, hA_mat_num_cols, hA_mat_nnz,
                                    dA_csrPtr, dA_cols, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, hB_mat_num_rows, hB_mat_num_cols, hB_mat_nnz,
                                    dB_csrPtr, dB_cols, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, hA_mat_num_rows, hB_mat_num_cols, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE,
                                    CUDA_R_32F) );
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, alg,
                                        spgemmDesc, &bufferSize1, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                        &alpha, matA, matB, &beta, matC,
                                        computeType, alg,
                                        spgemmDesc, &bufferSize1, dBuffer1) )

    CHECK_CUSPARSE(cusparseSpGEMM_getNumProducts(spgemmDesc, &num_prods) )

    // ask bufferSize3 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_estimateMemory(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, alg,
                                      spgemmDesc, chunk_fraction,
                                      &bufferSize3, NULL, NULL) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer3, bufferSize3) )
    
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_estimateMemory(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, alg,
                                      spgemmDesc, chunk_fraction,
                                      &bufferSize3, dBuffer3,
                                      &bufferSize2) )
    CHECK_CUDA( cudaFree(dBuffer3) ) // dBuffer3 can be safely freed to
                                     // save more memory

    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )

    // compute the intermediate product of A * B
    CHECK_CUSPARSE( 
        cusparseSpGEMM_compute(handle, opA, opB,
                                 &alpha, matA, matB, &beta, matC,
                                 computeType, alg,
                                 spgemmDesc, &bufferSize2, dBuffer2) )
    //--------------------------------------------------------------------------
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows, C_num_cols, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols,
                                        &C_nnz) )
    // allocate matrix C
    hC_mat_num_rows = (int)C_num_rows;
    hC_mat_num_cols = (int)C_num_cols;
    hC_mat_nnz = (uint64_t)C_nnz;

    CHECK_CUDA( cudaMalloc((void**) &dC_rows, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_cols, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )
    CHECK_CUDA( cudaMemset(dC_values, 0x0, C_nnz * sizeof(float)) )
    // fill dC_values if needed
    //--------------------------------------------------------------------------

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrPtr, dC_cols, dC_values) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, alg, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    Timing::stopTiming(true);
    //--------------------------------------------------------------------------
    // TODO: Change csr to coo
    // CHECK_CUSPARSE(
    //     cusparseXcsr2coo(handle, dC_csrPtr, hC_mat_nnz, hC_mat_num_rows, dC_rows,
    //                 CUSPARSE_INDEX_BASE_ONE)  )
    //--------------------------------------------------------------------------
    // device result load
    int* hC_rows_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    int* hC_columns_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    float* hC_values_tmp = new float[hC_mat_nnz];
    //TODO: change to coo
    // CHECK_CUDA( cudaMemcpy(hC_rows_tmp, dC_rows, hC_mat_nnz * sizeof(int),
    //                     cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_cols, hC_mat_nnz * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, hC_mat_nnz * sizeof(float),
                        cudaMemcpyDeviceToHost) )
    
    printCooToFile(outfile, hC_rows_tmp, hC_columns_tmp, hC_values_tmp, hC_mat_num_rows, hC_mat_num_cols, hC_mat_nnz);
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    CHECK_CUDA( cudaFree(dA_csrPtr) )
    CHECK_CUDA( cudaFree(dA_cols) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrPtr) )
    CHECK_CUDA( cudaFree(dB_cols) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrPtr) )
    CHECK_CUDA( cudaFree(dC_cols) )
    CHECK_CUDA( cudaFree(dC_values) )
    free(hC_rows_tmp);
    free(hC_columns_tmp);
    free(hC_values_tmp);

    // Stop timer
    Timing::flushTimer();
    // auto stop = high_resolution_clock::now();
    // auto end2end = duration_cast<nanoseconds>(stop - start);
    // std::cout << "END_TO_END_LATENCY, " << end2end.count() << std::endl;

    return EXIT_SUCCESS;
}


int main( int argc, char* argv[] )
{
    std::string      matrix1File( "" );
    std::string      matrix2File( "" );
    std::string      outfile( "" );

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--output" || arg == "-o" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if (arg == "--mat1" || arg == "-m1") {
            if (i < argc - 1) {
                matrix1File = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--mat2" || arg == "-m2") {
            if (i < argc - 1) {
                matrix2File = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else
        {
            std::cerr << "Unknown option " << arg << "\n";
            printUsageAndExit( argv[0] );
        }
    }

    try{
        #ifdef REUSE
            if (reuseCompute(matrix1File, matrix2File, outfile) == EXIT_FAILURE) {
                throw std::runtime_error("reuseCompute() returned EXIT_FAILURE");
            }
            // std::cerr << "Running Reuse" << "\n";
        #else
            if (compute(matrix1File, matrix2File, outfile) == EXIT_FAILURE) {
                throw std::runtime_error("compute() returned EXIT_FAILURE");
            }
            // std::cerr << "Running Compute" << "\n"; 
        #endif
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}