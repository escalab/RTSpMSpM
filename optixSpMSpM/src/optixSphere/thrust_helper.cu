#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

#include <thrust/count.h>
#include <thrust/unique.h>

#include <iostream> 
// #include <stdgpu/iterator.h>
// #include <thrust/get.h>
// #include <thrust/>
// #include <thrust/>
// #include <thrust/>
// #include <thrust/>

// this can't be in the main cpp file since the file containing cuda kernels to
// be compiled by nvcc needs to have .cu extensions. See here:
// https://github.com/NVIDIA/thrust/issues/614

void cuHelp () {
    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;
    H[1] = 20;
    H[2] = 38;
    H[3] = 46;
    
    // H.size() returns the size of vector H
    std::cout << "H has size " << H.size() << std::endl;

    // print contents of H
    for(size_t i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

    // resize H
    H.resize(2);
    
    std::cout << "H now has size " << H.size() << std::endl;

    // Copy host_vector H to device_vector D
    thrust::device_vector<int> D = H;
    
    // elements of D can be modified
    D[0] = 99;
    D[1] = 88;
    
    // print contents of D
    for(size_t i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    // H and D are automatically deleted when the function returns
    return;
}