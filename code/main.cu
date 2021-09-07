/**
 * @author : Samundra Karki
 * @work: Bachlor's thesis implmentation
 * @title: GPU-parallel preconditioner for Approximated Krylov-based kernel ridge regression training 
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <stdint.h>

#include "fetch_data.h"
#include "kernel_ridge_regression.h"
#include "gauss_preconditioner.cuh"
#include "main_preconditioner.h"
#include "cublas_v2.h"
#include "generateSyntheticData.h"

//Here you can uncomment a preconditioner defination to use that preconditioner. Note you must comment other definations of the preconditioner
#define UNPRECONDITIONED
// #define TRI_DIAG_PRECONDITIONER
// #define LOCAL_PRECONDITIONER

//Uncomment the dataset that you want to use. The bigger datasets are underprocess so, please use only synthetic data for now.
#define SYNTHETIC_DATA
// #define ENERGY_EFFICIENCY_DATA

//Uncomment the size of the data set you want to use.
#ifdef SYNTHETIC_DATA
// const int data_size = 15;

const int data_size = 1000;
// const int data_size = 3000;
// const int data_size = 5000;
// const int data_size = 7000;
// const int data_size = 9000;
// const int data_size = 11000;

const int testing_data_size = 30;

const int input_dim = 3;
const int output_dim = 1;
#endif

#ifdef ENERGY_EFFICIENCY_DATA

const int data_size = 768;
const int input_dim = 8;
const int output_dim = 2;
const int r = 600;
const int bits = 8;

#endif

#ifdef QM9

const int data_size = 133885;
const int input_dim = 11960;
const int output_dim = 1;

#endif

#ifdef MD17

const int data_size = 993237;
const int input_dim = 3121;
const int output_dim = 1;

#endif
extern void gauss_seidel_preconditioner(double* jacobi_preconditioner_matrix, double* kernel_matrix, int size);
void print_data(double* matrix, int data_size, int dim);
void symmetrise_preconditioner(double* preconditioner, int data_size);

void checkCUDAErrorMain(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

__global__ void build_identity_matrix(double* indetityMat, int data_size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= data_size) return;

    indetityMat[idx*data_size+idx] = 1;
}


int main()
{
    //host arrays
    double* input_data; ///<Input data
    double* output_data; ///<Output data
    double* kernel_matrix; ///<Kernel matrix
    double* preconditioner; ///<Preconditioner
    double* hIdMat; ///<Identity matrix
    double* testing_data; ///<testing data
    double* testing_output; ///<true actual output data
    double* predicted_output; ///< predicted output


    //device array
    double* dIdentityMatrix; ///< Identity matrix

    //Host memory allocation
    input_data = (double* ) malloc(data_size * input_dim*sizeof(double));  //input_data_allocation
    output_data = (double* ) malloc(data_size * output_dim*sizeof(double));  //output_data_allocation
    testing_data = (double*) malloc(testing_data_size* input_dim*sizeof(double));
    testing_output= (double*) malloc(testing_data_size* output_dim*sizeof(double));
    predicted_output = (double*) malloc(testing_data_size* output_dim*sizeof(double));
    kernel_matrix = (double* ) malloc(data_size * data_size*sizeof(double));  //kernel_matrix_allocation
    //calloc initilalizes the value of the matrix to zero.
    preconditioner = (double* ) calloc(data_size * data_size, sizeof(double)); //preconditioner allocation
    hIdMat = (double*) calloc(data_size*data_size, sizeof(double)); //identity matrix allocation
    
    //Device Memory allocation
    cudaMalloc((void**)&dIdentityMatrix, data_size*data_size*sizeof(double));
    checkCUDAErrorMain("cudaMalloc.: 1");
    
    //Constructing Identity matrix of size (datasize*datasize)
    cudaMemcpy(dIdentityMatrix, hIdMat, data_size*data_size*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorMain("MemCpy");
    
    int block_size = 1024;
    int grid_size = (data_size + (block_size - 1)) / block_size;

    build_identity_matrix<<<grid_size, block_size>>>(dIdentityMatrix, data_size);
    cudaDeviceSynchronize();    
    cudaMemcpy(hIdMat, dIdentityMatrix, data_size*data_size*sizeof(double), cudaMemcpyDeviceToHost);
    checkCUDAErrorMain("MemCpy");
    
    //Finish construction of identity matrix

    //GENERATE/LOAD training data and testing data
    #ifdef SYNTHETIC_DATA
        generateData(input_data, testing_data, output_data, testing_output, testing_data_size, data_size, input_dim, output_dim);
    #endif

    #ifdef ENERGY_EFFICIENCY_DATA
        fetch_data_energy_efficiency(input_data, output_data, data_size, input_dim, output_dim);
    #endif

    #ifdef QM9
        fetch_data_QM9(input_data, output_data, data_size, input_dim, output_dim);
    #endif

    #ifdef MD17
        fetch_data_MD17(input_data, output_data, data_size, input_dim, output_dim);
    #endif

    
    // printf("The input data:\n");
    // print_data(input_data, data_size, input_dim);
    // printf("The output data:\n");
    // print_data(output_data, data_size, output_dim);
    
    // printf("The testing input data:\n");
    // print_data(testing_data, testing_data_size, input_dim);
    // printf("The testing output data:\n");
    // print_data(testing_output, testing_data_size, output_dim);
    
    // printf("The global kernel matrix:\n");
    // build_matrix(input_data, kernel_matrix, data_size, input_dim);
    // print_data(kernel_matrix, data_size, data_size);
    
    #ifdef TRI_DIAG_PRECONDITIONER
        build_matrix(input_data, kernel_matrix, data_size, input_dim);
        // print_data(kernel_matrix, data_size, data_size);
        // printf("\n");
        regularize_kernel_matrix(kernel_matrix,  hIdMat, data_size);
        // print_data(kernel_matrix, data_size, data_size);
        gauss_seidel_preconditioner(preconditioner, kernel_matrix, data_size);
        // print_data(preconditioner, data_size, data_size);

        kernel_ridge_regression(kernel_matrix, preconditioner, input_data, output_data, hIdMat, testing_data, predicted_output, testing_data_size,  data_size, input_dim, output_dim);
        // print_data(predicted_output, testing_data_size, output_dim);
    #endif

    #ifdef LOCAL_PRECONDITIONER
        // Uncomment the section to choose the size of the neighborhood,
        int r = 5;
        // int r = 55;
        // int r = 105;

        
        const int bits = 20;
        double* sorted_inData; ///<Sorted input data
        double* sorted_outData; ///<Sorted output data
        
        sorted_inData = (double* ) malloc(data_size * input_dim*sizeof(double)); //sorted input data allocation
        sorted_outData = (double* ) malloc(data_size * output_dim*sizeof(double)); //sorted output data allocation
        // for(int r=1 ; r<150; r++) { 
           
            printf("r=%i p=%i\n", r, data_size);
            main_preconditioner(input_data, output_data, sorted_inData, sorted_outData, preconditioner, data_size, input_dim, r, bits);   
            
            // print_data(sorted_inData, data_size, input_dim);
            // print_data(sorted_outData, data_size, output_dim);
            
            // printf("The symmetric preconditioner matrix:\n");
            symmetrise_preconditioner(preconditioner, data_size);
            // print_data(preconditioner, data_size, data_size);

            build_matrix(sorted_inData, kernel_matrix, data_size, input_dim);
            regularize_kernel_matrix(kernel_matrix, hIdMat, data_size);
            // print_data(kernel_matrix, data_size, data_size);
            kernel_ridge_regression(kernel_matrix, preconditioner, sorted_inData, sorted_outData, hIdMat, testing_data, predicted_output, testing_data_size,  data_size, input_dim, output_dim);
        // }
        free(sorted_inData);
        free(sorted_outData);
    #endif

    #ifdef UNPRECONDITIONED
        build_matrix(input_data, kernel_matrix, data_size, input_dim);
        // print_data(kernel_matrix, data_size, data_size);
        regularize_kernel_matrix(kernel_matrix,  hIdMat, data_size);
        // print_data(kernel_matrix, data_size, data_size);
        kernel_ridge_regression(kernel_matrix, hIdMat, input_data, output_data, hIdMat, testing_data, predicted_output, testing_data_size,  data_size, input_dim, output_dim);
        // print_data(predicted_output, data_size, data_size);

    #endif

    

    free(input_data);
    free(output_data);
    free(kernel_matrix);
    free(preconditioner);
    free(hIdMat);
    free(testing_data);
    free(testing_output);
    free(predicted_output);
    cudaFree(dIdentityMatrix);
}

/** \brief Function to print passed matrix.
    \param matrix Matrix to be printed 
    \param data_size Number of rows of the matrix
    \param dim Number of columns of the matrix
    \return void
*/
void print_data(double* matrix, int data_size, int dim)
{
      for(int i = 0; i < data_size; i++){
        for(int j=0; j< dim; j++){
            printf("%f ", matrix[j*data_size+i]);
        }
        printf("\n");
    }
}

/** \brief Function to make the matrix symmtric.
    \param preconditioner Matrix to be symmtrised 
    \param data_size Number of rows/column of the matrix
    \return void
*/
void symmetrise_preconditioner(double* preconditioner, int data_size)
{
    double* d_preconditioner;
    double* d_preconditioner_symm;

    cudaMalloc((void**)&d_preconditioner, data_size * data_size * sizeof(double));
    checkCUDAErrorMain("Main: CudaMalloc: 2.");

    cudaMalloc((void**)&d_preconditioner_symm, data_size * data_size * sizeof(double));
    checkCUDAErrorMain("Main: CudaMalloc: 3.");

    
    cudaMemcpy(d_preconditioner, preconditioner,data_size*data_size*sizeof(double), cudaMemcpyHostToDevice);
    
    cublasStatus_t stat;
    cublasHandle_t handle;
    double alpha = 0.5f;
    double beta = 0.5f;

    stat = cublasCreate(&handle);

    // P = 0.5(P + P^t)
    stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, data_size, data_size, 
        &alpha, d_preconditioner, data_size, 
        &beta, d_preconditioner, data_size, d_preconditioner_symm, data_size);
    
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error");
    }

    cudaMemcpy(preconditioner, d_preconditioner_symm, data_size*data_size* sizeof(double), cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
    cudaFree(d_preconditioner);
    cudaFree(d_preconditioner_symm);

}


