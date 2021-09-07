#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "gauss_preconditioner.cuh"

#define BATCH_SIZE 1

__global__ void gauss_seidel_preconditioner_construction(double* kernel_matrix, double* preconditioner, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        preconditioner[idx*size+idx] = kernel_matrix[idx*size+idx];
        if(idx > 0) preconditioner[(idx*size+idx)-1] = kernel_matrix[(idx*size+idx)-1];
        if(idx < (size-1)) preconditioner[(idx*size+idx)+1] = kernel_matrix[(idx*size+idx)+1];     
    }
}

void gauss_seidel_preconditioner(double* preconditioner, double* kernel_matrix, int dataSize)
{

    double* device_kernel_matrix;
    double* device_preconditioner;

    cudaMalloc((void**)&device_kernel_matrix, dataSize*dataSize*sizeof(*kernel_matrix));
    cudaMalloc((void**)&device_preconditioner, dataSize*dataSize*sizeof(*preconditioner));

    cudaMemcpy(device_kernel_matrix, kernel_matrix, dataSize*dataSize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_preconditioner, preconditioner, dataSize*dataSize*sizeof(double), cudaMemcpyHostToDevice);
    
    int block_size = 1024;
    int grid_size = (dataSize + (block_size - 1)) / block_size;
    
    gauss_seidel_preconditioner_construction<<<grid_size, block_size>>>(device_kernel_matrix, device_preconditioner, dataSize);
    cudaDeviceSynchronize();

    // cudaMemcpy(preconditioner, device_preconditioner, dataSize*dataSize*sizeof(double), cudaMemcpyDeviceToHost);

    cublasHandle_t handle;
    cublasStatus_t cublas_status;
    cublas_status = cublasCreate(&handle);
    // cublasErrorCheck(cublas_status);

    double* hostPtrPrecondMatrix[BATCH_SIZE]; ///<host ptr to preconditioner matrix
    double* hostPtrInvPrecondMat[BATCH_SIZE]; ///<host ptr to inverse of preconditioner matrix
    double** dPtrPrecondMatrix; ///<pointer to preconditioner matrix
    double* dInvPrecondMat; ///< device inverse of preconditioner matrix
    double** dPtrInvPrecondMat; ///< device pointer to inverse of preconditioner matrix


    int* pivotArray; ///< pivot array in LU decomposition P*A = L*U
    int* infoArray; ///< stores the inforamtion of the operation i.e. if singular or not
    int h_InfoArray = 0; ///< host variable to store the information


    cudaMalloc((void**)&(dPtrPrecondMatrix), BATCH_SIZE*sizeof(double*));
    cudaMalloc((void**)&(dInvPrecondMat), BATCH_SIZE*dataSize*dataSize*sizeof(double)); 
    cudaMalloc((void**)&(dPtrInvPrecondMat), BATCH_SIZE*sizeof(double*));

    cudaMalloc(&pivotArray, dataSize * BATCH_SIZE * sizeof(int));
    cudaMalloc(&infoArray, BATCH_SIZE * sizeof(int));

    hostPtrPrecondMatrix[0] = device_preconditioner;
    hostPtrInvPrecondMat[0] = dInvPrecondMat;
    
    cudaMemcpy(dPtrPrecondMatrix, hostPtrPrecondMatrix, BATCH_SIZE*sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(dPtrInvPrecondMat, hostPtrInvPrecondMat, BATCH_SIZE*sizeof(double*), cudaMemcpyHostToDevice);


    cublas_status = cublasDgetrfBatched(handle, dataSize, dPtrPrecondMatrix, dataSize, pivotArray, infoArray, BATCH_SIZE);
    if(cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error Occured");
    }
    cudaMemcpy(&h_InfoArray, infoArray, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<BATCH_SIZE; i++)
    {
        if (h_InfoArray != 0) {
            fprintf(stderr, "%d Factorization of matrix Failed: Matrix may be singular\n", h_InfoArray);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }

    cublas_status = cublasDgetriBatched(handle, dataSize, dPtrPrecondMatrix, dataSize, pivotArray, dPtrInvPrecondMat, dataSize, infoArray, BATCH_SIZE);
    if(cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error Occured");
    }   
    cudaMemcpy(&h_InfoArray, infoArray, BATCH_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<BATCH_SIZE; i++)
    {
        if (h_InfoArray != 0) {
            fprintf(stderr, "%d Factorization of matrix Failed: Matrix may be singular\n", h_InfoArray);
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }
    cudaMemcpy(preconditioner, dInvPrecondMat, dataSize*dataSize*sizeof(double), cudaMemcpyDeviceToHost);
}


