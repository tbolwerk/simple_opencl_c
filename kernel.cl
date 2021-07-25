__kernel void addition(__global const uint *A, __global const uint *B, __global uint *C){
    int i = get_global_id(0);

    C[i] = A[i] + B[i];
}