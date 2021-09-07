#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "nearest_neighbors.cuh"
#include "local_input_system.cuh"
#include "local_kernel_matrix.cuh"
#include "local_linear_system.cuh"
#include "construct_preconditioner.cuh"
#include "main_preconditioner.h"


cudaEvent_t startMP, stopMP; 
float millisecondsMP;


#define TIME_startMP {cudaEventCreate(&startMP); cudaEventCreate(&stopMP); cudaEventRecord(startMP);}
#define TIME_stopMP(a) {cudaEventRecord(stopMP); cudaEventSynchronize(stopMP); cudaEventElapsedTime(&millisecondsMP, startMP, stopMP); printf("%s: Elapsed time: %lf ms\n", a, millisecondsMP); }

void checkCUDAErrorMP(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

void print_matrix_dev_input(double* d_matrix, int nRows, int nColumns)
{
    double* host_arr; ///<Host array that will store the device matrix 
    host_arr = (double*) malloc(nRows*nColumns*sizeof(double));

	cudaMemcpy(host_arr, d_matrix, nRows*nColumns*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<nRows; i++)
    {
        for(int j=0; j<nColumns; j++)
        {
            printf("%f ", host_arr[j*nRows+i]);
        }
        printf("\n");
    }

    free(host_arr);
}

void print_matrix_dev_kernel(double* d_matrix, int nRows, int nColumns)
{
    double* host_arr; ///<Host array that will store the device matrix 
    host_arr = (double*) malloc(nRows*nColumns*sizeof(double));

	cudaMemcpy(host_arr, d_matrix, nRows*nColumns*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i=0; i<nRows; i++)
    {
        for(int j=0; j<nColumns; j++)
        {
            printf("%f ", host_arr[i*nColumns+j]);
        }
        printf("\n");
    }

    free(host_arr);
}

void print_map_index(uint64_t* dMappingIndex, int dataSize, int r)
{
    uint64_t* hMappingIndex; //< Mapping indices 
    hMappingIndex = (uint64_t*) malloc(dataSize*r*sizeof(uint64_t));
    cudaMemcpy(hMappingIndex, dMappingIndex, dataSize*r*sizeof(double), cudaMemcpyDeviceToHost);
    
    for(int j=0; j<dataSize; j++){
        for(int i=0; i<r; i++)
        {
            printf("%li ", hMappingIndex[j*r+i]);
        }
        printf("\n");
    }

    free(hMappingIndex);
    
}

void print_matrix_host(double* host_arr, int nRows, int nColumns)
{
    for(int i=0; i<nRows; i++)
    {
        for(int j=0; j<nColumns; j++)
        {
            printf("%f ", host_arr[j*nRows+i]);
        }
        printf("\n");
    }
}



void print_sorted_data(double** dsortedDataPoints, int dataSize, int dim)
{
    double** hSorted = new double*[dim];
    for(int i=0; i<dim; i++)
    {
        hSorted[i] = (double*)malloc(dataSize*sizeof(double));
    }
    for(int i=0; i<dim; i++)
    {
        cudaMemcpy(hSorted[i], dsortedDataPoints[i], dataSize*sizeof(double), cudaMemcpyDeviceToHost);
    }
    for (int p=0; p<dataSize; p++)
	{
		for (int d=0; d<dim; d++)
		{
            printf("%f ", hSorted[d][p]);
        }
        printf("\n");
    }

    for(int i=0; i<dim; i++)
    {
        free(hSorted[i]); 
    }
    delete[] hSorted;
    
}

void main_preconditioner(double* inputData, double* outputData, double* sorted_inData, double* sorted_outData, 
                            double* preconditioner, int dataSize, int dim, int r, int bits)
{
    double* dOutputData; //<Device output data
    double* dSorted_outData; //<Device input data
    double** dsortedDataPoints = new double*[dim]; ///<sorted data points
    double* localInputData; ///<local input data formed from r nearest neigbors
    double* localKernelMatrix; ///<local kernel matrix formed from r nearest neighbors
    double* localSolution;  ///<solution of local lonear system formed from r nearest neighbors
    double* dPrecondtioner; ///< device preconditioner matrix
    uint64_t* dMappingIndex; //< Mapping indices 
    uint64_t* order;
    
    cudaMalloc((void**)&order, dataSize*sizeof(uint64_t));
    checkCUDAErrorMP("MP: cudaMalloc 1...");
    
    for(int i=0; i<dim; i++)
    {
        cudaMalloc((void**)&(dsortedDataPoints[i]), dataSize*sizeof(double));
        checkCUDAErrorMP("MP: cudaMalloc 1.");
    }
    cudaMalloc((void**)&(localInputData), dataSize*dim*r*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 2.");
	cudaMalloc((void**)&localKernelMatrix, dataSize*r*r*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 3.");
	cudaMalloc((void**)&localSolution, dataSize*r*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 4.");
    cudaMalloc((void**)&dPrecondtioner, dataSize*dataSize*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 5.");
    cudaMalloc((void**)&dMappingIndex, dataSize*r*sizeof(uint64_t));
    checkCUDAErrorMP("MP: cudaMalloc 6.");
    cudaMalloc((void**)&(dOutputData), dataSize*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 7.");
    cudaMalloc((void**)&(dSorted_outData), dataSize*sizeof(double));
    checkCUDAErrorMP("MP: cudaMalloc 8.");

    cudaMemcpy( dOutputData, outputData, dataSize*sizeof(double),cudaMemcpyHostToDevice);
    checkCUDAErrorMP("MP: ....");

    // TIME_startMP
    compute_nearest_neighbors(inputData, dOutputData, dsortedDataPoints, sorted_inData, dSorted_outData, order, dataSize, dim, bits, r);
    cudaMemcpy(sorted_outData, dSorted_outData,  dataSize*sizeof(double),cudaMemcpyDeviceToHost);
    // TIME_stopMP("Sort the data")
    // print_matrix_host(sorted_inData, dataSize, dim);
    // print_map_index(order, dataSize, 1);
    // print_matrix_host(outputData, dataSize, 1);
    // print_sorted_data(dsortedDataPoints, dataSize, dim);
    checkCUDAErrorMP("MP: Nearest neighbors.");

    // TIME_startMP
    compute_local_input_data(localInputData, dsortedDataPoints, dMappingIndex, dataSize, dim, r);
    // TIME_stopMP("Compute the local data points and mapping indices")   
    // print_matrix_dev_input(localInputData, dataSize*r, dim);
    // print_map_index(dMappingIndex, dataSize, r);
    checkCUDAErrorMP("MP: Memcpy 2.");


    // TIME_startMP
    // printf("The local kernel matrix: \n");
    compute_local_kernel_matrix(localKernelMatrix, localInputData, dim, dataSize, r);
    // TIME_stopMP("Compute Local kernel Matrix")
    // print_matrix_dev_kernel(localKernelMatrix, dataSize, r*r);
    checkCUDAErrorMP("MP: Memcpy 3.");

    // TIME_startMP
    // printf("The local solutions: \n");
    solve_local_linear_system(localKernelMatrix, localSolution, dataSize, r);
    // TIME_stopMP("Solve local Linear system")
    // print_matrix_dev_kernel(localSolution, dataSize, r);
    checkCUDAErrorMP("MP: Memcpy 4.");
    
    cudaMemcpy(dPrecondtioner, preconditioner, dataSize*dataSize*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorMP("MP: Memcpy 5.");
    
    // printf("The non symmetric preconditioner: \n");
    // TIME_startMP
    construct_preconditioner(dPrecondtioner, localSolution, dMappingIndex, order, dataSize, r);
    // TIME_stopMP("Map the local linear system solution to preconditioner")
    // print_matrix_dev_kernel(dPrecondtioner, dataSize, dataSize);
    
    cudaMemcpy(preconditioner, dPrecondtioner, dataSize*dataSize*sizeof(double), cudaMemcpyDeviceToHost);
    checkCUDAErrorMP("MP: Memcpy 2.");
    // print_matrix_host(preconditioner, dataSize, dataSize);

    cudaFree(localInputData);
    cudaFree(order);
    cudaFree(dOutputData);
    cudaFree(dSorted_outData);
    cudaFree(dMappingIndex);
    cudaFree(dPrecondtioner);
    cudaFree(localSolution);
    cudaFree(localKernelMatrix);
    for(int i=0; i<dim; i++)
    {
        cudaFree(dsortedDataPoints[i]);
    }
    checkCUDAErrorMP("MP: Free mem.");

    delete[] dsortedDataPoints;
}

// int main()
// {
//     double * input_data;
//     double* preconditioner;
    
//     int data_size = 10;
// 	int input_dim = 3;
// 	int bits = 20;
//     int r = 3;

//     input_data = (double* ) malloc(data_size * input_dim*sizeof(double));
//     preconditioner = (double*) calloc(data_size* data_size, sizeof(double));

//     // print_data(input_data, data_size, input_dim);

//     main_preconditioner(input_data, preconditioner, data_size, input_dim, r, bits);
	

//     free(input_data);
//     free(preconditioner);
// }

