May 21st, 2024
==============
Launch
-----
- We are launching 1-dimension ray, corresponding to the number of entries in mat1

Data Transfer
-----
- Current Output buffer:
    --------------------
    - Dimension is of `#_of_mat1_entry * #_of_mat2_entry`
    - So it allows storing as following:

    `hit_data->result[(int)ray_idx.x * hit_data->matrix1size + (int)sphere_idx] = resultFloat;`
    - where it basically is `mat1_entry_idx * #_of_mat1_entry + mat2_entry_idx` (a * dimension + b)
    - Each would store a float3 with final location i, j and result z
- Expectation Output:
    -----------------
    - Assume we are multipling mat_1 <sub>ixj</sub> with mat_2 <sub>jxk</sub>
    - New Dimension should be `i * k`
    - We should change the sbt binding table for output buffer
        - In resultBufferSetup function
        - This is furthermore setted up in hg_sbt.data.result = state.d_result;
    - Because the actual calculation is in parallel, how do we map the result with the new buffer?

    ./bin/optixSphere -m1 "MatrixSphere/494_bus.mtx" -m2 "MatrixSphere/494_bus.mtx"