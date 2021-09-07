#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"

cudaEvent_t startLLS, stopLLS; 
float millisecondsLLS;


#define TIME_startLLS {cudaEventCreate(&startLLS); cudaEventCreate(&stopLLS); cudaEventRecord(startLLS);}
#define TIME_stopLLS(a) {cudaEventRecord(stopLLS); cudaEventSynchronize(stopLLS); cudaEventElapsedTime(&millisecondsLLS, startLLS, stopLLS); printf("%s: Elapsed time: %lf ms\n", a, millisecondsLLS); }


__global__ void compute_output_vector(double* outputVector, int r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>=r) return;
    
    outputVector[idx] = 0;
    outputVector[0] = 1;
}

void checkCUDAErrorLLS(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

void cublasErrorCheck(cublasStatus_t cublasStat)
{
    if(cublasStat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error Occured");
    }
}

void solve_local_linear_system(double* localKernelMatrix, double* localSolution, int dataSize, int r)
{
    double* localSolution_i; ///< local solution for ith data point
    double* dOutput;  ///< local output vector for ith data point

    cudaMalloc((void**)&(localSolution_i), r*sizeof(double));
    checkCUDAErrorLLS("LLS: cudaMalloc 1");
    cudaMalloc((void**)&(dOutput), r*sizeof(double));
    checkCUDAErrorLLS("LLS: cudaMalloc 2");

    int blockSize = 1024;
    int gridSize = (r+(blockSize-1))/blockSize;
    compute_output_vector<<<gridSize, blockSize>>>(dOutput, r);
    cudaDeviceSynchronize();
    checkCUDAErrorLLS("LLS: Computing output vector for local linear system.");

    cublasHandle_t handle;
    cublasStatus_t cublas_status;
    cublas_status = cublasCreate(&handle);
    cublasErrorCheck(cublas_status);

    const int BATCH_SIZE = dataSize;
    
    double* hostPtrKernelMatrix[BATCH_SIZE]; ///<host ptr to kernel matrix
    double* hostPtrInvKernelMat[BATCH_SIZE]; ///<host ptr to inverse of kernel matrix
    double* dKernelMatrix;  ///< device kernel matrix
    double** dPtrKernelMatrix; ///<pointer to kernel matrix
    double* dInvKernelMat; ///< device inverse of kernel matrix
    double** dPtrInvKernelMat; ///< device pointer to inverse of kernel matrix

    int* pivotArray; ///< pivot array in LU decomposition P*A = L*U
    int* infoArray; ///< stores the inforamtion of the operation i.e. if singular or not
    int* h_InfoArray; ///< host variable to store the information

    double al = 1.0f;
    double bet = 0.0f;

    const double* alpha = &al;
    const double* beta = &bet;

    cudaMalloc((void**)&(dKernelMatrix), BATCH_SIZE*r*r*sizeof(double)); 
    checkCUDAErrorLLS("LLS: cudaMalloc 2");

    // cudaMalloc((void**)&(dKernelMatrix), r*r*sizeof(double)); 
    cudaMalloc((void**)&(dPtrKernelMatrix), BATCH_SIZE*sizeof(double*));
    checkCUDAErrorLLS("LLS: cudaMalloc 3");

    cudaMalloc((void**)&(dInvKernelMat), BATCH_SIZE*r*r*sizeof(double)); 
    checkCUDAErrorLLS("LLS: cudaMalloc 4");

    cudaMalloc((void**)&(dPtrInvKernelMat), BATCH_SIZE*sizeof(double*));
    checkCUDAErrorLLS("LLS: cudaMalloc 5");

    
    cudaMalloc(&pivotArray, r * BATCH_SIZE * sizeof(int));
    checkCUDAErrorLLS("LLS: cudaMalloc 6");

    cudaMalloc(&infoArray, BATCH_SIZE * sizeof(int));
    checkCUDAErrorLLS("LLS: cudaMalloc 7");

    h_InfoArray = (int*) malloc(BATCH_SIZE * sizeof(int));
    
    for(int i=0; i<BATCH_SIZE; i++)
    {
        // cudaMemcpy(dKernelMatrix, localKernelMatrix+(i*(r*r)), r*r*sizeof(double), cudaMemcpyDeviceToDevice);
        hostPtrKernelMatrix[i] = localKernelMatrix+(i*(r*r));
        hostPtrInvKernelMat[i] = dInvKernelMat+(i*(r*r));
    }

    cudaMemcpy(dPtrKernelMatrix, hostPtrKernelMatrix, BATCH_SIZE*sizeof(double*), cudaMemcpyHostToDevice);
    checkCUDAErrorLLS("LLS: Memcpy 1");

    cudaMemcpy(dPtrInvKernelMat, hostPtrInvKernelMat, BATCH_SIZE*sizeof(double*), cudaMemcpyHostToDevice);
    checkCUDAErrorLLS("LLS: Memcpy 2");

    TIME_startLLS
    // LU decomposition of kernel matrix
    cublas_status = cublasDgetrfBatched(handle, r, dPtrKernelMatrix, r, pivotArray, infoArray, BATCH_SIZE);
    TIME_stopLLS("LU Factorization local matrices")
    cublasErrorCheck(cublas_status);
    cudaMemcpy(h_InfoArray, infoArray, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorLLS("LLS: Memcpy 3");
    
    for(int i=0; i<BATCH_SIZE; i++)
    {
        if (h_InfoArray[i] != 0) {
            fprintf(stderr, "%d -- %i Factorization of matrix Failed: Matrix may be singular\n", h_InfoArray[i], i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }
    
    TIME_startLLS
    cublas_status = cublasDgetriBatched(handle, r, dPtrKernelMatrix, r, pivotArray, dPtrInvKernelMat, r, infoArray, BATCH_SIZE);
    TIME_stopLLS("Inverse of the local matrices")
    cublasErrorCheck(cublas_status);    
    cudaMemcpy(h_InfoArray, infoArray, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    checkCUDAErrorLLS("LLS: Memcpy 4");

    for(int i=0; i<BATCH_SIZE; i++)
    {
        if (h_InfoArray[i] != 0) {
            fprintf(stderr, "%d -- %i Factorization of matrix Failed: Matrix may be singular\n", h_InfoArray[i], i);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }
    
    for(int i=0; i<BATCH_SIZE; i++)
    {
        cublas_status = cublasDgemv(handle, CUBLAS_OP_N, r, r, alpha, dInvKernelMat+(i*r*r), r, dOutput, 1, beta, localSolution_i, 1);
        cublasErrorCheck(cublas_status);
        cudaMemcpy(localSolution+(i*r), localSolution_i, r*sizeof(double),cudaMemcpyDeviceToDevice);
        checkCUDAErrorLLS("LLS: Memcpy 5");

    }

    cudaFree(localSolution_i);
    cudaFree(dOutput);
    cudaFree(dKernelMatrix);
    cudaFree(dInvKernelMat);
    cudaFree(pivotArray);
    cudaFree(infoArray);
}
