#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <device_launch_parameters.h>
#include "cublas_v2.h"
#include "kernel_ridge_regression.h"
#include "preconditioned_conjugate_gradient.cuh"

#define SIGMA 2
#define LAMBDA 0.0000000001

cudaEvent_t startKRR, stopKRR; 
float millisecondsKRR;


#define TIME_startKRR {cudaEventCreate(&startKRR); cudaEventCreate(&stopKRR); cudaEventRecord(startKRR);}
#define TIME_stopKRR(a) {cudaEventRecord(stopKRR); cudaEventSynchronize(stopKRR); cudaEventElapsedTime(&millisecondsKRR, startKRR, stopKRR); printf("%s: Elapsed time: %lf ms\n", a, millisecondsKRR); }



extern void preconditioned_conjugate_gradient(double* kernel_matrix, double* preconditioner, double* output_data, 
    double* approximated_solution, int data_size, int output_dim);

void checkCUDAErrorKRR(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

int iDivUp_global(int hostPtr, int b)
{
	 return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}


//CPU Implementation of building a kernel matrix
// double l1_norm(double* input_data, int idx_i, int idx_j, int data_size, int dim)
// {
//     double temp_array[dim];
    
//     double* x_i;
//     x_i = (double* ) malloc( dim * sizeof(double));

//     double* x_j;
//     x_j = (double* ) malloc( dim * sizeof(double)); 

//     double l1_norm = 0;
    
//     for(int i = 0; i < dim; i++)
//     {
//         x_i[i] = input_data[i*data_size+idx_i];
//     }

//     for(int i = 0; i < dim; i++)
//     {
//         x_j[i] = input_data[i*data_size+idx_j];
//     }

//     // for(int i = 0; i < dim; i++)
//     // {
//     //     printf("%f %f \n", x_i[i], x_j[i]);
//     // }
//     for(int i = 0; i < dim; i++)
//     { 
//         temp_array[i] = abs(x_i[i]) - abs(x_j[i]);
//         // printf("%f ", temp_array[i]);
//         l1_norm += temp_array[i];
//     }

//     free(x_i);
//     free(x_j);
//     return l1_norm;
// }

//GPU kernel function to build a kernel matrix
__global__ void build_kernel_matrix(double* input_data, double* kernel_matrix, int data_size, int dim)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if(tidx<data_size && tidy < data_size){
		double norm = 0;
		double kernel_value =0;
		for(int d=0; d<dim; d++)
		{
			norm += pow(input_data[tidx+(d*data_size)]-input_data[tidy+(d*data_size)],2);
		}
		norm = sqrt(norm);
		kernel_value = exp(-pow(norm,2)/SIGMA);
		kernel_matrix[tidx+data_size*tidy] = kernel_value;
		// printf("----%i, %i, %f, %f \n",tidx, tidy, norm, kernel_value);
	}
}

//The comented part is the CPU implementation of build a kernel matrix
void build_matrix(double* input_data, double* kernel_matrix, int data_size, int dim)
{
    // double l1_norm_value;    
    // for(int i = 0; i < data_size; i++)
    // {
    //     for (int j = 0; j < data_size; j++)
    //     {
    //         l1_norm_value = l1_norm(input_data, i, j, data_size, dim);
    //         kernel_matrix[i*data_size+j] = exp(-(abs(l1_norm_value)/SIGMA));
    //     }
    // }
    double* dInputData;
    double* dKernelMatrix;

    cudaMalloc((void**)&dInputData, data_size*dim*sizeof(double));
    cudaMalloc((void**)&dKernelMatrix, data_size*data_size*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMemcpy(dInputData, input_data, data_size*dim*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorKRR("Memcpy");

    dim3 block_size(32 ,32);
	dim3 grid_size(iDivUp_global(data_size, 32) ,iDivUp_global(data_size, 32));

    build_kernel_matrix<<<grid_size, block_size>>>(dInputData, dKernelMatrix, data_size, dim);
    cudaDeviceSynchronize();
    checkCUDAErrorKRR("Building kernel matrix");

    cudaMemcpy(kernel_matrix, dKernelMatrix, data_size*data_size*sizeof(double), cudaMemcpyDeviceToHost);
    checkCUDAErrorKRR("cudaMemcpy");
}


void print_approximated_Solution(double* hWeight, int data_size, int output_dim)
{
    for(int i=0; i<output_dim; i++)
    {
        for(int j=0; j<data_size; j++)
        {
            printf("%f ", hWeight[i*data_size+j]);
        }
        printf("\n");
    }
}

void regularize_kernel_matrix(double* kernel_matrix, double* hIdMat, int data_size)
{
    double* dKernelMatrix;
    double* dRegularizedKernelMatrix;
    double* didentityMatrix;

    cudaMalloc((void**)&dKernelMatrix, data_size*data_size*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMalloc((void**)&didentityMatrix, data_size*data_size*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMalloc((void**)&dRegularizedKernelMatrix, data_size*data_size*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMemcpy(didentityMatrix, hIdMat, data_size*data_size*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorKRR("Memcpy");

    cudaMemcpy(dKernelMatrix, kernel_matrix, data_size*data_size*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorKRR("Memcpy");

    cublasStatus_t stat;
    cublasHandle_t handle;
    const double lambda = 0.0000000001;
    double alpha = 1.0f;
    double beta = lambda;
    
    stat = cublasCreate(&handle);
    

    stat = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, data_size, 
        &alpha, dKernelMatrix, data_size, 
        &beta, didentityMatrix, data_size, dRegularizedKernelMatrix, data_size);
       
    
    if(stat != CUBLAS_STATUS_SUCCESS){
        printf("Error: Error status is %i", stat);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    cublasDestroy(handle);
    cudaMemcpy(kernel_matrix, dRegularizedKernelMatrix, data_size*data_size* sizeof(double), cudaMemcpyDeviceToHost);

}

__global__ void compute_kernel_values(double* training_points, double* kernel_values, double* evaluation_point, int dim, int size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= size) return;
    
    double norm = 0;
    for(int d=0; d<dim; d++)
	{
		norm += pow(training_points[idx+(d*size)]-evaluation_point[d],2);
	}
	norm = sqrt(norm);
	kernel_values[idx] = exp(-pow(norm,2)/SIGMA);
}

void predict_output(double* weights, double* testing_data, double* training_data, double* predicted_output, int testing_data_size, int data_size, int input_dim, int output_dim)
{
    double* dWeights;
    double** dTestingData;
    double* dTrainingData;
    double* dkernelVector;
    double* dPredictedOutput;
    


    dTestingData = new double*[testing_data_size];
	for (int d = 0; d < testing_data_size; d++)
	{
		cudaMalloc((void**)&(dTestingData[d]), input_dim*sizeof(double));
		checkCUDAErrorKRR("cudaMalloc");
	}

    cudaMalloc((void**)&dWeights, data_size*output_dim*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMalloc((void**)&dTrainingData, data_size*input_dim*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMalloc((void**)&dkernelVector, data_size*sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMalloc((void**)&dPredictedOutput, sizeof(double));
    checkCUDAErrorKRR("cudaMalloc");

    cudaMemcpy(dTrainingData, training_data, data_size*input_dim*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorKRR("Memcpy.");

    cudaMemcpy(dWeights, weights, data_size*output_dim*sizeof(double), cudaMemcpyHostToDevice);
    checkCUDAErrorKRR("Memcpy");

     //CUBLAS -- status
    cublasStatus_t stat;
     // //CUBLAS context -- handle
    cublasHandle_t handle;
 
 //  //Create context
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error in context creatiuon");
    }
   
    double alpha = 1.0f;
    double beta = 0.0f;

    int block_size = 1024;
    int grid_size = (data_size * (block_size-1))/block_size;

    double* currentEvaluationPoint;
    currentEvaluationPoint = (double*) malloc(input_dim*sizeof(double));

    
    for(int i=0; i<testing_data_size; i++)
    {
        for(int j=0; j<input_dim; j++)
        {
            currentEvaluationPoint[j] = testing_data[j*testing_data_size+i];
        }
        cudaMemcpy(dTestingData[i], currentEvaluationPoint, input_dim*sizeof(double), cudaMemcpyHostToDevice);

        compute_kernel_values<<<grid_size, block_size >>>(dTrainingData, dkernelVector, dTestingData[i], input_dim, data_size);
        cudaDeviceSynchronize();

    
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, output_dim, data_size, &alpha, dkernelVector, data_size, dWeights, data_size, &beta, dPredictedOutput, output_dim);
        if(stat != CUBLAS_STATUS_SUCCESS)
        {
            printf("Error");
        }
        cudaMemcpy( predicted_output+i, dPredictedOutput, sizeof(double) ,cudaMemcpyDeviceToHost);
    }

    
    cudaFree(dWeights);
    cudaFree(dTrainingData);
    for(int i=0; i<testing_data_size; i++){
        // cudaFree(dTestingData[i]);
    }
    delete [] dTestingData;
    cudaFree(dkernelVector);
    cudaFree(dPredictedOutput);
    cublasDestroy(handle);
}

void kernel_ridge_regression(double* kernel_matrix, double* preconditioner, double* input_data,  double* output_data, double* hIdMat, double* testing_data,
                              double* predicted_output, int testing_data_size, int data_size, int input_dim, int output_dim)
{
    //Host array
    double* hWeight;
    hWeight = (double*) calloc(data_size*output_dim, sizeof(double));
   
    //Conjugate gradient method
    if(output_dim == 1) {
        preconditioned_conjugate_gradient(kernel_matrix, preconditioner, output_data, hWeight, data_size, 1);
    }
    
    if(output_dim > 1)
    {
        double* dimWise_output;
        dimWise_output = (double*) malloc(data_size*1* sizeof(double));

        double* d_output;
        cudaMalloc((void**)&d_output, data_size*output_dim*sizeof(double));

        cudaMemcpy(d_output, output_data, data_size*output_dim*sizeof(double), cudaMemcpyHostToDevice);
        for(int d=0; d<output_dim; d++){
            cudaMemcpy(dimWise_output, d_output + (d*data_size), data_size*sizeof(double), cudaMemcpyDeviceToHost);
            preconditioned_conjugate_gradient(kernel_matrix, preconditioner, dimWise_output, hWeight+(d*data_size), data_size, 1);
            printf("\n");
        }
    }
    // Uncomment the below code to generate the tesing output data
    // 
    // predict_output(hWeight, testing_data, input_data, predicted_output, testing_data_size, data_size, input_dim, output_dim);
    //    

    free(hWeight);
}
