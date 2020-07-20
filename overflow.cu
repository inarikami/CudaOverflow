#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef MIN
#define MIN(a,b) (a > b ? a : b)
#endif


__device__ unsigned int block_idx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
        threadIdx.y * (x = threads->x) +
        threadIdx.z * (x *= threads->y) +
        blockIdx.x * (x *= threads->z) +
        blockIdx.y * (x *= blocks->z) +
        blockIdx.z * (x *= blocks->y);
}

__device__ void gpu_mergesort_up(long* a, int* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || a[i] < a[j])) {
            dest[k] = a[i];
            i++;
        }
        else {
            dest[k] = a[j];
            j++;
        }
    }
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    long idx = block_idx(threads, blocks);
    long start = width * idx * slices, middle, end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;
        middle = MIN(start + (width >> 1), size);
        end = MIN(start + width, size);
        gpu_mergesort_up(source, dest, start, middle, end);
        start += width;
    }
}

void mergesort(long* a, long size, dim3 threads_per_block, dim3 blocks_per_grid) {
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    checkCudaErrors(cudaMalloc((void**)&D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**)&D_swp, size * sizeof(long)));

    checkCudaErrors(cudaMemcpy(D_data, a, size * sizeof(int), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

    checkCudaErrors(cudaMemcpy(D_threads, &threads_per_block, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocks_per_grid, sizeof(dim3), cudaMemcpyHostToDevice));


    long nThreads = threads_per_block.x * threads_per_block.y * threads_per_block.z *
        blocks_per_grid.x * blocks_per_grid.y * blocks_per_grid.z;


    long* A = D_data;
    long* B = D_swp;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads)*width) + 1;

        gpu_mergesort<<<blocks_per_grid, threads_per_block>>>(A, B, size, width, slices,
            D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }
    checkCudaErrors(cudaMemcpy(a, A, size*sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
}


// 100% GPU grown cage-free all natural 
int main(int argc, char** argv)
{
    int devID;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char**)argv);

    //Get GPU information
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
        devID, props.name, props.major, props.minor);

    dim3 threads_per_block;
    dim3 blocks_per_grid;

    threads_per_block.x = 32;
    threads_per_block.y = 1;
    threads_per_block.z = 1;

    blocks_per_grid.x = 16;
    blocks_per_grid.y = 1;
    blocks_per_grid.z = 1;

    int a[] = { 5,9,1,3,4,6,6,3,2 };
    unsigned int len = sizeof(a) / sizeof(int);

    mergesort(a, size, threads_per_block, blocks_per_grid);

    for (int i = 0; i < len; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    return EXIT_SUCCESS;
}