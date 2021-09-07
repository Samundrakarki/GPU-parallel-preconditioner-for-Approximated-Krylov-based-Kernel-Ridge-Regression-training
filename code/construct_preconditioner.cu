#include <stdio.h>
#include "construct_preconditioner.cuh"

cudaEvent_t startCP, stopCP; 
float millisecondsCP;


#define TIME_startCP {cudaEventCreate(&startCP); cudaEventCreate(&stopCP); cudaEventRecord(startCP);}
#define TIME_stopCP(a) {cudaEventRecord(stopCP); cudaEventSynchronize(stopCP); cudaEventElapsedTime(&millisecondsCP, startCP, stopCP); printf("%s: Elapsed time: %lf ms\n", a, millisecondsCP); }



__global__ void preconditioner_construction(double* preconditioner, double* localSolution, uint64_t* mappingIndex, uint64_t* order, int dataSize, int r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=(dataSize*r)) return;
    
    int idx_a = idx/r;

    int loc = (int)mappingIndex[idx];
    int global_loc = (int) order[loc];
    // printf("%i, %i, %i, %i, %i \n", idx, idx_a, loc, global_loc, (idx_a*dataSize)+global_loc);
    preconditioner[(idx_a*dataSize)+global_loc] = localSolution[idx];
}

void construct_preconditioner(double* dpreconditioner, double* dLocalSolution, uint64_t* dMappingIndex, uint64_t* order, int dataSize, int r)
{
    int blockSize = 512;
    int gridSize = (dataSize*r + (blockSize - 1)) / blockSize;
    TIME_startCP
    preconditioner_construction<<<gridSize, blockSize>>>(dpreconditioner, dLocalSolution, dMappingIndex, order, dataSize, r);
    cudaDeviceSynchronize();
    TIME_stopCP("Map preconditioner")
}
