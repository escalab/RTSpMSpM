//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <iostream>

#include "optixSpMSpM.h"
// #include <cuda/helpers.h>
#include <vector>

#include <sutil/vec_math.h>
#include <nvtx3/nvToolsExt.h>
// #include <stdgpu/unordered_map.cuh>
// #include <stdgpu/iterator.h>        // device_begin, device_end
// #include <cuco/static_map.cuh>

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
        )
{
    unsigned int p0, p1, p2;
    p0 = __float_as_uint( prd->x );
    p1 = __float_as_uint( prd->y );
    p2 = __float_as_uint( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = __uint_as_float( p0 );
    prd->y = __uint_as_float( p1 );
    prd->z = __uint_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            __uint_as_float( optixGetPayload_0() ),
            __uint_as_float( optixGetPayload_1() ),
            __uint_as_float( optixGetPayload_2() )
            );
}

// static "C"  void checkSphere()
// { (is false)
//     {
//         return false;
//     }
    
//     return true;
    
// }

#if defined(NOTHING)
extern "C" __global__ void __raygen__rg()
{
    int dx = 1;
    // printf("NOTHING%d\n", dx);
}
#else
extern "C" __global__ void __raygen__rg()
{
    int dx = 1;
    #if defined(ARCHSUP)
            // printf("ARCHSUP%d\n", dx);
        #elif defined(NOMEM)
            // printf("Nomem%d\n", dx);
        #elif defined(NOINT)
            // printf("NOINT%d\n", dx);
        #else
            // printf("ATOMIC%d\n", dx);
    #endif

    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    int ray_idx = idx.x;
    // the thread block is one-d
    // [t0 t1 t2 t3] ray_idx = idx.x
    // [t0 t1 t2 t3] [t0 t1 t2 t3] [t0 t1 t2 t3] ray_idx = idx.x + dim.x * block_idx.x
    

    RayData* ray_data = reinterpret_cast<RayData*>(optixGetSbtDataPointer());
    
    if (idx.x < ray_data->size && idx.y == 0)
    { 
        // for (int i = 0; i < ray_data->size; ++i) 
        // {
        //     printf("%f ", ray_data->originVec[i]);
        // }
        const float origin_k = ray_data->originVec[ray_idx].y;
        const float3 origin = make_float3(origin_k, 0.0f, 0.0f);
        const float3 direction = make_float3( origin_k, 1e16f, 0.0f);

        // printf("Going from %f to %f\n", origin_k, direction.y);
        // printf("Ray gen from: %.3f, %.3f, %.3f, %.3f\n", origin.x, origin.y, direction.x, direction.y);

        float3 payload_rgb = ray_data->originVec[ray_idx];
        #if defined(NOINT)
            // This will make sure nothing is hit since radius is .1, apart by 1, also going in z direction since nothing is there
            float origin_j = 0.5;     
            float3 origin_i = make_float3(origin_j, 0.0f, 1.0f);
            float3 direction_i = make_float3( origin_j, 0.0f, 1e16f);
            trace(params.handle,
                origin_i,
                direction_i,
                0.00f, // tmin
                1e16f, // tmax
                &payload_rgb);

            // Simulate hit
            // printf("Simulate %d /5 number of hits", ray_data->size);
            for( int i = 0; i < ray_data->size; i += 5){
                const uint3 ray_idx = optixGetLaunchIndex();
                // const unsigned int           sphere_idx    = optixGetPrimitiveIndex();
                // const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
                // const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

                float4 sphere;
                // sphere center (q.x, q.y, q.z), sphere radius q.w
                // optixGetSphereData( gas, sphere_idx, sbtGASIndex, 0.f, &sphere );
                float3    payload = ray_data->originVec[ray_idx.x];

                // SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
                // float sphereData   = hit_data->sphereColor[sphere_idx];
                float resultFloat = payload.z * 3.33 * i; // Random float

                // Store data in result buffer
                int numRow = ray_data->size;
                int numCol = ray_data->size;
                uint64_t rowIndex = (uint64_t)payload.x;
                // uint64_t colIndex = (uint64_t)sphere.y;
                uint64_t idx_x = rowIndex * numCol + 90000 + i; // Random int
                uint64_t largestIdx = numRow * numCol + idx_x;
                // printf("Address of hit_data->result[%d]: %p\n", idx, &(hit_data->result[idx]));
                // atomicAdd(&(hit_data->result[idx]), resultFloat); //IMPORTNAT
            }
        #else
            trace(params.handle,
                origin,
                direction,
                0.00f, // tmin
                1e16f, // tmax
                &payload_rgb);
        #endif
        //params.image[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
    }
            
}
#endif



extern "C" __global__ void __miss__ms()
{
    // MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    // float3    payload = getPayload();
    // setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ) );
}


#if defined(NOTHING)
extern "C" __global__ void __anyhit__ch()
{
    int idx = 1;
}
#else
extern "C" __global__ void __anyhit__ch()
{
    #if defined(ARCHSUP)
        int idx = optixGetPrimitiveIndex() & 0xff;
        SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
        // atomicAdd(&(hit_data->result[optixGetPrimitiveIndex()]), 1);
        hit_data->result[idx] += 1;
        optixIgnoreIntersection();
    #elif defined(NOMEM)
        optixIgnoreIntersection();
    #elif defined(NOINT)
        int idx = 1;
    #else
        // Normal Atomic
        const uint3 ray_idx = optixGetLaunchIndex();
        const unsigned int           sphere_idx    = optixGetPrimitiveIndex();
        const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
        const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
        // printf("Sphere Index %d \n", sphere_idx);

        float4 sphere;
        // sphere center (q.x, q.y, q.z), sphere radius q.w
        optixGetSphereData( gas, sphere_idx, sbtGASIndex, 0.f, &sphere );
        float3    payload = getPayload();

        SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
        float sphereData   = hit_data->sphereColor[sphere_idx];
        float resultFloat = payload.z * sphereData;

        // Store data in result buffer
        int numRow = hit_data->resultNumRow;
        int numCol = hit_data->resultNumCol;
        uint64_t rowIndex = (uint64_t)payload.x;
        uint64_t colIndex = (uint64_t)sphere.y;
        uint64_t idx = rowIndex * numCol + colIndex;
        uint64_t largestIdx = numRow * numCol;
        // printf("Address of hit_data->result[%d]: %p\n", idx, &(hit_data->result[idx]));
        if (idx < largestIdx) {
            atomicAdd(&(hit_data->result[idx]), resultFloat);
        }
        optixIgnoreIntersection();
    #endif
    // CSV formatted printf
    // Format: ray_idx.x,ray_idx.y,ray_idx.z,
    //         payload.x,payload.y,payload.z,
    //         sphere_idx,sphere.x,sphere.y,sphere.z,
    //         sphereData,resultFloat,
    // Meaning: ray_idx, 0, 0, 
    //         mat_1_x (result_row), mat_1_y, mat_1_value,
    //         sphere_idx, mat_2_x, mat_2_y (result_col), 0,
    //         mat_2_value, mat_1 val * mat_2 val
    // printf("%d,%d,%d,%.0f,%.0f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
    //            (int)ray_idx.x, (int)ray_idx.y, (int)ray_idx.z, 
    //            payload.x, payload.y, payload.z, 
    //            (int)sphere_idx, sphere.x, sphere.y, sphere.z, 
    //            sphereData, resultFloat);

    // printf("%d\n", (int)ray_idx.x );
    // printf("%.3f\n", payload.z * sphereData);
    // printf("Mat_1 [%.0f][%.0f](%.3f) * Mat_2 [%.0f][%.0f](%.3f) -> Mat_result[%.0f][%.0f](%.3f)\n", payload.x, payload.y, payload.z, sphere.x, sphere.y, sphereData, payload.x, sphere.y, resultFloat);
    // printf("Current Result[%.0f][%.0f],(%.3f)\n", payload.x, sphere.y, hit_data->result[idx]);
}
#endif


