#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/thrust_helper.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

OPTIX_add_sample_executable( optixSpMSpM target_name
  optixSpMSpM.cu
  optixSpMSpM.cpp
  optixSpMSpM.h
  Sphere.cpp
  Sphere.h
  Util.h
  OPTIONS -rdc true
  )

# target_compile_features(${target_name} INTERFACE cxx_std_17 cuda_std_17)

# target_include_directories(${target_name} PRIVATE ${STDGPU_INCLUDE_DIR})

#target_compile_options(${target_name} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
#      --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)

# Link your project against stdgpu

# target_compile_features(${target_name} INTERFACE cxx_std_17 cuda_std_17)

# target_compile_options(${target_name} PRIVATE --compiler-options=-Wall --compiler-options=-Wextra
        # --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler -Wno-subobject-linkage)

# target_compile_options(${target_name} PUBLIC --expt-extended-lambda --expt-relaxed-constexpr)
target_compile_options(${target_name} PUBLIC -Wno-subobject-linkage) 
target_link_libraries( ${target_name} 
    ${CUDA_LIBRARIES}
    # cuco
    # stdgpu::stdgpu
)

# if(USE_CUDA)
#     add_definitions(-DUSE_CUDA)
# endif()
# include("/nfshome/hoz006/OptixSDK/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/build2/_deps/cuco-src/include/cuco/dynamic_map.cuh")
