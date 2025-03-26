include(FindPackageHandleStandardArgs)

set(stdgpu_DIR "${CMAKE_SOURCE_DIR}/stdgpu/bin/lib/cmake/stdgpu" CACHE PATH "Path to stdgpu installed location.")

find_library(STDGPU_LIB 
    NAMES stdgpu
    PATHS "${CMAKE_SOURCE_DIR}/stdgpu/bin/lib/cmake/stdgpu"
)
find_path(stdgpu_INCLUDE_DIR NAMES "${stdgpu_DIR}/../libstdgpu.a")
find_package_handle_standard_args(stdgpu REQUIRED_VARS STDGPU_LIB)

if (stdgpu_FOUND)
  mark_as_advanced(STDGPU_LIB)
endif()