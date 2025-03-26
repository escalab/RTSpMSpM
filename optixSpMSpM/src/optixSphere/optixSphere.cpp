/**
 * To Compile:
 *    Run command "make" in the build folder
 * To run: 
 * ./bin/optixSphere -m1 "../../../Tool/PythonTool/output/spmGen10000_13300.mtx" -m2 "../../../Tool/PythonTool/output/spmGen10000_13300.mtx”
 * ./bin/optixSphere -m1 "../../../Tool/PythonTool/output/spmGen16384_268435.mtx" -m2 "../../../Tool/PythonTool/output/spmGen16384_268435.mtx" -o "./test123.mtx"
 * ./bin/optixSphere -m1 "../../../Tool/PythonTool/output/spmGen32768_1073741.mtx" -m2 "../../../Tool/PythonTool/output/spmGen32768_1073741.mtx" -o "./test123.mtx"
 * ./bin/optixSphere -m1 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" -m2 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" -o "./test.mtx"
 * ./bin/optixSphere -m1 "./MatrixSphere/test1.mtx" -m2 "./MatrixSphere/test2.mtx" -o "./test12345.mtx"
 * ./bin/optixSphere -m1 "../../../../trace/sparse_matrices/suitSparse/all/patents_main/patents_main.mtx" -m2 "../../../../trace/sparse_matrices/suitSparse/all/patents_main/patents_main.mtx" -o "./test12345.mtx"
 * 
 * ./bin/optixSphere -m1 "MatrixSphere/494_bus.mtx" -m2 "MatrixSphere/494_bus.mtx" -o "./test123.mtx"
 * Nsys command profiling:
 * nsys nvprof /home/OptixSDK/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/build/bin/optixSphere_Atomic -m1 "../../../Tool/PythonTool/output/spmGen32768_1073741.mtx" -m2 "../../../Tool/PythonTool/output/spmGen32768_1073741.mtx” -o "./temp.mtx"
 * ../../../Tool/PythonTool/output/spmGen16384_268435.mtx
*/

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/sutil.h>

#include "optixSphere.h"

#include "Sphere.h"
#include <sutil/Timing.h>
#include <iomanip>
#include <iostream>
#include <string>

#include <map>
#include <nvtx3/nvToolsExt.h>

// #include <cuco/static_map.cuh>
// #include <stdgpu/cstddef.h>   
// #include <stdgpu/unordered_map.cuh>

#include <chrono> 
using namespace std::chrono;
#include "Util.h"

typedef SbtRecord<RayData>      RayDataRec;
typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<SphereData>   SphereDataRec;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --output | -o <filename>    Specify file for image output\n";
    std::cerr << "         --mat1 | -m1 <filename>     Specify file for matrix 1\n";
    std::cerr << "         --mat2 | -m2 <filename>     Specify file for matrix 2\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

/**
 * Load in Rays from matrix 1 data file
 */
void mat1ToGPU( const std::string& filePath, optixState& state )
{
    Timing::startTiming("M1 File Read");
    float3 *output, *d_output;
    std::vector<float3> tempMatrix;

    std::string fileName = sutil::sampleDataFilePath( filePath.c_str() );

    std::ifstream file(fileName);
    std::string line;
    std::ifstream input( fileName.c_str(), std::ios::in );
    SUTIL_ASSERT_MSG( input.is_open(), "Unable to open " + fileName + "." );

    bool isFirstDataLine = true;
    int rows, cols;
    uint64_t nonZeros;

    while (std::getline(file, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '%') {
            continue;
        }

        if (isFirstDataLine) {
            // Read the first data line containing dimensions
            std::istringstream iss(line);
            if (!(iss >> rows >> cols >> nonZeros)) {
                std::cerr << "Error reading matrix dimensions." << std::endl;
                return;
            }

            //Finalize Result buffer
            state.m_result_dim = std::make_pair(state.m_result_dim.first,cols);
            std::cout << "THE RESULT MATRIX IS " << state.m_result_dim.first << " by " << state.m_result_dim.second << std::endl;
            isFirstDataLine = false;
            continue;
        }
        
        std::istringstream iss(line);
        float x, y, val;
        
        if (!(iss >> x >> y >> val)) {
            // Handle parsing error
            continue; 
        }

        tempMatrix.push_back(make_float3(x - 1,y - 1,val)); // covnert to 0-based
    }
    Timing::stopTiming(true);

    // Allocate space for matrix
    Timing::startTiming("M1 Data Transfer");
    // nvtxRangePushA("Mat1Upload");
    output = tempMatrix.data();
    uint64_t cnt = tempMatrix.size();
    CUDA_CHECK(cudaMalloc(&d_output, cnt*sizeof(float3)));

    cudaMemcpy(d_output, output, cnt*sizeof(float3), cudaMemcpyHostToDevice);

    state.d_size = cnt;
    state.matrixFloat = output;
    state.d_matrix = d_output;
    // nvtxRangePop();
    Timing::stopTiming(true);
}

/**
 * Load in Sphere from matrix 2 data file
 */
void storeSphereData( optixState& state, std::string matrixFile2 )
{
    // nvtxRangePushA("Mat2Upload");
    Timing::startTiming("M2 Data Load");
    //
    // Matrix File Input
    //
    Timing::startTiming("M2 File Read");
    std::string sphereFileName = sutil::sampleDataFilePath( matrixFile2.c_str() );
    Sphere        sphere(state.context, sphereFileName );
    Timing::stopTiming(true);

    // nvtxRangePushA("M2 Data Transfer");
    float *output, *d_output;
    std::vector<float> sphereValues = sphere.value();
    uint64_t cnt = sphere.points().size();

    // Allocate space for matrix
    output = sphereValues.data();
    std::cout << "OUTPUT CNT:" << cnt << std::endl;
    CUDA_CHECK(cudaMalloc(&d_output, cnt*sizeof(float)));
    cudaMemcpy(d_output, output, cnt*sizeof(float), cudaMemcpyHostToDevice);
    state.sphere_size = cnt;
    state.spherePoints = d_output;
    // nvtxRangePop();

    state.devicePoints = 0;
    state.deviceRadius = 0;

    state.m_result_dim = sphere.printDim();  //Fill in result buffer (state.m_result_dim.first, dummy value)

    // nvtxRangePushA("M2 Coordinates Transfer");      //TODO: WE want the gpu both sparse and dense matrix to convert to this in cuda cores and then calculate the overhead
    createOnDevice( sphere.points(), &state.devicePoints );
    // nvtxRangePop();
    // nvtxRangePushA("M2 Radius Transfer");
    createOnDevice( sphere.radius(), &state.deviceRadius );
    // nvtxRangePop();
    //
    //
    //
    // nvtxRangePop();
    Timing::stopTiming(true);
}

void resultBufferSetup( optixState& state )
{
    float* buf;
    state.d_result_buf_size = state.m_result_dim.first * state.m_result_dim.second * sizeof(float);
    CUDA_CHECK(cudaMalloc( &buf, state.d_result_buf_size ));
    cudaMemset(buf, 0, state.d_result_buf_size);
    state.d_result = buf;
}

// Initialize CUDA and create OptiX context
void contextSetUp(optixState& state)
{
    Timing::startTiming("Context Setup");
    state.context = nullptr;
    
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    // nvtxRangePushA("Context Setup");

    CUcontext cuCtx = 0;  // zero means take the current context
    // nvtxRangePushA("optixInit");
    OPTIX_CHECK( optixInit() );
    // nvtxRangePop();
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    // nvtxRangePushA("createContext");
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &state.context ) );
    // nvtxRangePop();
    // nvtxRangePop();
    Timing::stopTiming(true);
}

void buildGAS(optixState& state)
{
    //
    // accel handling
    //
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    
    
    // sphere build input
    OptixBuildInput sphere_input = {};
    uint64_t numVert = state.sphere_size;
    
    Timing::startTiming("buildGAS");
    // nvtxRangePushA("buildGAS");


    sphere_input.type                      = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    sphere_input.sphereArray.numVertices   = numVert;
    sphere_input.sphereArray.vertexBuffers = &state.devicePoints;
    sphere_input.sphereArray.vertexStrideInBytes = sizeof(float3);
    sphere_input.sphereArray.radiusBuffers = &state.deviceRadius;
    sphere_input.sphereArray.radiusStrideInBytes   = sizeof( float );

    uint32_t sphere_input_flags[1]            = {OPTIX_GEOMETRY_FLAG_NONE};
    sphere_input.sphereArray.flags         = sphere_input_flags;
    sphere_input.sphereArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    
    // nvtxRangePushA("optixAccelComputeMemoryUsage");
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &sphere_input, 1, &gas_buffer_sizes ) );
    // nvtxRangePop();
    CUdeviceptr d_temp_buffer_gas;
    // nvtxRangePushA("mallocGASbuf");
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );
    // nvtxRangePop();

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                            compactedSizeOffset + 8 ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    // nvtxRangePushA("optixAccelBuild");
    OPTIX_CHECK( optixAccelBuild( state.context,                //To+ODO: Detailed analysis for each function
                                    0,  // CUDA stream
                                    &accel_options, &sphere_input,
                                    1,  // num build inputs
                                    d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                    d_buffer_temp_output_gas_and_compacted_size, gas_buffer_sizes.outputSizeInBytes, &state.gas_handle,
                                    &emitProperty,  // emitted property list
                                    1               // num emitted properties
                                    ) );
    // nvtxRangePop();

    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    // nvtxRangePushA("copyBackCompact");
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
    // nvtxRangePop();

    // nvtxRangePushA("optixAccelCompact");
    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
    // nvtxRangePop();
    // nvtxRangePop();

    Timing::stopTiming(true);
}

void createModule (optixState& state)
{
    // nvtxRangePushA("Create Module");
    Timing::startTiming("create module");
    //
    // Create module
    //
    OptixModuleCompileOptions module_compile_options = {};
    #if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    #endif

    state.pipeline_compile_options.usesMotionBlur                   = false;
    state.pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues                 = 3;
    state.pipeline_compile_options.numAttributeValues               = 1;
    state.pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

    size_t      inputSize = 0;
    // nvtxRangePushA("getInputData");
    const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixSphere.cu", inputSize );
    // nvtxRangePop();

    // nvtxRangePushA("optixModuleCreate");
    OPTIX_CHECK_LOG( optixModuleCreate( state.context, &module_compile_options, &state.pipeline_compile_options, input,
                                        inputSize, LOG, &LOG_SIZE, &state.module ) );
    // nvtxRangePop();

    OptixBuiltinISOptions builtin_is_options = {};

    // nvtxRangePushA("optixBuiltinISModuleGet");
    builtin_is_options.usesMotionBlur      = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK_LOG( optixBuiltinISModuleGet( state.context, &module_compile_options, &state.pipeline_compile_options,
                                                &builtin_is_options, &state.sphere_module ) );
    // nvtxRangePop();
    //
    //
    //
    Timing::stopTiming(true);
    // nvtxRangePop();
}

void createProgramGroups (optixState& state)
{
    // nvtxRangePushA("Create Program Groups");
    Timing::startTiming("create program groups");
    //
    // Create program groups
    //
    OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
    raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module            = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                state.context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &state.raygen_prog_group
                ) );

    OptixProgramGroupDesc miss_prog_group_desc  = {};
    miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module            = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                state.context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &state.miss_prog_group
                ) );

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleAH            = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleCH            = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS            = state.sphere_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
                state.context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                LOG, &LOG_SIZE,
                &state.hit_prog_group
                ) );
    //
    //
    //
    Timing::stopTiming(true);
    // nvtxRangePop();
}

void createPipeline ( optixState& state )
{
    // nvtxRangePushA("Create Pipeline");
    Timing::startTiming("create pipeline");
    //
    // Link pipeline
    //
    const uint32_t    max_trace_depth  = 1;
    OptixProgramGroup program_groups[] = { state.raygen_prog_group, state.miss_prog_group, state.hit_prog_group };
    
    state.pipeline_link_options.maxTraceDepth            = max_trace_depth;
    OPTIX_CHECK_LOG( optixPipelineCreate(
                state.context,
                &state.pipeline_compile_options,
                &state.pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                LOG, &LOG_SIZE,
                &state.pipeline
                ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                0,  // maxCCDepth
                                                0,  // maxDCDEpth
                                                &direct_callable_stack_size_from_traversal,
                                                &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
    //
    //
    //
    Timing::stopTiming(true);
    // nvtxRangePop();
}

void createSbt ( optixState& state )
{
    // nvtxRangePushA("Create SBT");
    Timing::startTiming("create sbt");
    //
    // Set up shader binding table
    //  
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof( RayDataRec );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );

    RayDataRec rg_sbt;
    rg_sbt.data.originVec = state.d_matrix;
    rg_sbt.data.size    = state.d_size;
    
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( raygen_record ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );
    

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof( MissSbtRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
    MissSbtRecord ms_sbt;
    ms_sbt.data = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK( optixSbtRecordPackHeader( state.miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( miss_record ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );


    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = sizeof( SphereDataRec );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
    SphereDataRec hg_sbt;
    hg_sbt.data.sphereColor = state.spherePoints;
    resultBufferSetup( state );
    hg_sbt.data.result = state.d_result;    // result buffer pntr
    hg_sbt.data.resultNumRow = state.m_result_dim.first;
    hg_sbt.data.resultNumCol = state.m_result_dim.second;
    hg_sbt.data.matrix1size = state.d_size;
    hg_sbt.data.matrix2size = state.sphere_size;


    OPTIX_CHECK( optixSbtRecordPackHeader( state.hit_prog_group, &hg_sbt ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( hitgroup_record ),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
                ) );

    state.sbt.raygenRecord                = raygen_record;
    state.sbt.missRecordBase              = miss_record;
    state.sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof( SphereDataRec );
    state.sbt.hitgroupRecordCount         = 1;
    //
    //
    //
    Timing::stopTiming(true);
    // nvtxRangePop();
}


void printResult ( optixState& state , std::string fileName)
{
    // nvtxRangePushA("Result Download");

    // Open File
    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return;
    }

    // Copy result to host
    float* result = (float*)malloc(state.d_result_buf_size);
    cudaMemcpy(result, state.d_result, state.d_result_buf_size, cudaMemcpyDeviceToHost);

    std::map<std::pair<int, int>, float> resultMatrix;

    int numRows = state.m_result_dim.first;
    int numCols = state.m_result_dim.second;

    // nvtxRangePop();

    // nvtxRangePushA("Accumulate Result");

    // MTX header
    outFile << "%%MatrixMarket matrix coordinate real general\n";
    // Dimensions and non-zero count
    outFile << numRows << " " << numCols << " "  << "\n"; // TODO: nnz enable function: << nonZeroCount

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            // Calculate the 1D index
            uint64_t idx = i * numCols + j;
            if(result[idx] == 0){
                continue;
            }
            // Store the element in the map with (row, column) as the key
            // resultMatrix[std::make_pair(i, j)] = result[idx];
            outFile << i + 1 << " " << j + 1 << " " << result[idx] << "\n";
            // printf("Result: (%d)(%d)[%f]\n", i+1, j+1, result[idx]);
        }
    }

    // TODO: Calculate result nnz size, and output it to file
    uint64_t nonZeroCount = resultMatrix.size();
    outFile.seekp(std::ios::beg); // Move to the beginning
    outFile << "%%MatrixMarket matrix coordinate real general\n";
    outFile << numRows << " " << numCols << " " << nonZeroCount << "\n";

    // nvtxRangePop();

    outFile.close();
}

int main( int argc, char* argv[] )
{
    optixState state;

    // Matrix Input File
    // Selection: 494_bus 662_bus test1 test2 dw256B dwb512
    // std::string      matrix1File( "MatrixSphere/dw256B.mtx" );
    // std::string      matrix2File( "MatrixSphere/dwb512.mtx" );
    std::string      matrix1File( "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" );
    std::string      matrix2File( "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" );
    std::string      outfile( "MatrixSphere/result.mtx" );
    
    int             width  = 2000;
    int             height = 2000;
    state.width            = width;
    state.height           = height;

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
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
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
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        #if defined(ARCHSUP)
            std::cout << "ARCHSUP" << std::endl;
        #elif defined(NOMEM)
            std::cout << "NOMEM" << std::endl;
        #elif defined(NOINT)
            std::cout << "NOINT" << std::endl;
        #elif defined(NOTHING)
            std::cout << "NOTHING" << std::endl;
        #else
            std::cout << "ATOMIC" << std::endl;
        #endif
        // Start timer
        Timing::reset();
        auto start = high_resolution_clock::now();
        Timing::startTiming("total time");
        
        // Matrix Load to GPU
        storeSphereData(state, matrix2File);    // Mat2
        mat1ToGPU(matrix1File, state);

        Timing::startTiming("computation time no io");
        contextSetUp( state );
        buildGAS ( state );
        createModule ( state );
        createProgramGroups ( state );
        createPipeline ( state );
        createSbt ( state );

        // nvtxRangePushA("Launch");
        Timing::startTiming("launch");
        //
        // Launch
        //
        CUstream stream;
        CUDA_CHECK( cudaStreamCreate( &stream ) );
        state.params.handle       = state.gas_handle;
        CUdeviceptr d_param;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_param ),
                    &state.params, sizeof( state.params ),
                    cudaMemcpyHostToDevice
                    ) );
        // printf("SIZE PARAMS: %d, %d", sizeof(state.params), sizeof(Params));

        uint64_t numRayLaunch = state.d_size;
        OPTIX_CHECK( optixLaunch( state.pipeline, stream, d_param, sizeof( Params ), &state.sbt, numRayLaunch, /*height=*/1, /*depth=*/1 ) );
        // nvtxRangePop();

        cudaStreamSynchronize(stream);
        Timing::stopTiming(true);
        Timing::stopTiming(true);

        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
        //
        //
        //
        CUDA_SYNC_CHECK();
        
        
        //
        // Display results
        //
        Timing::startTiming("retrieve result and print");
        printResult(state, outfile);
        Timing::stopTiming(true);
        Timing::stopTiming(true);

        printf("success!");
        
        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer    ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_matrix               ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.spherePoints           ) ) );

            OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( state.hit_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( state.miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( state.raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( state.module ) );
            OPTIX_CHECK( optixModuleDestroy( state.sphere_module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );
        }
        // Stop timer
        Timing::flushTimer();
        // auto stop = high_resolution_clock::now();
        // auto end2end = duration_cast<nanoseconds>(stop - start);  ;
        // std::cout << "END_TO_END_LATENCY = " << end2end.count() << std::endl;
        CUDA_SYNC_CHECK();
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
