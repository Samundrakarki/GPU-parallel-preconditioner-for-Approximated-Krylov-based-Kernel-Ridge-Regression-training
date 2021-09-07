#include <stdlib.h>
#include <stdio.h>
#include <curand.h>

void checkCUDAErrorGenerate(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

__global__ void compute_output(double* outputData, double* inputData, int inputDim, int data_size)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= data_size) return;

    outputData[idx] = sinf(inputData[idx]);
}


void generateData(double* input_data, double* testing_data, double* output_data, double* testingOutput, int testing_data_size , int data_size, int input_dim, int output_dim)
{
    double* dInputData; ///< device input data
    double* dOutputData; ///< device output data

    double* dTestingData; ///<device testing data
    double* dTestingOutput; ///<device testing data

    //device mem allocation
    cudaMalloc((void**)&dInputData, data_size * input_dim*sizeof(double));
    checkCUDAErrorGenerate("CudaMalloc: 4.");

    cudaMalloc((void**)&dOutputData, data_size * output_dim*sizeof(double));
    checkCUDAErrorGenerate("CudaMalloc: 5.");

    cudaMalloc((void**)&dTestingData, testing_data_size * input_dim*sizeof(double));
    checkCUDAErrorGenerate("CudaMalloc: 4.");

    cudaMalloc((void**)&dTestingOutput, testing_data_size * output_dim*sizeof(double));
    checkCUDAErrorGenerate("CudaMalloc: 5.");

    //generating random numbers
    curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	for (int d = 0; d < input_dim; d++ )
	{
        curandGenerateUniformDouble(gen, dInputData+(d*(data_size)), data_size);
        //Uncomment following code generate the testing data
        // 
        // curandGenerateUniformDouble(gen, dTestingData+(d*(testing_data_size)), testing_data_size);
        // 
    }  
    checkCUDAErrorGenerate("Generating random numbers");

    //computing output as dOuput[i] = sin(x_i), x_i is the first dimension of the input data
    int block_size = 1024;
    int grid_size = (data_size + (block_size-1))/ block_size;

    compute_output<<<grid_size, block_size >>>(dOutputData, dInputData, input_dim, data_size);
    cudaDeviceSynchronize();
    checkCUDAErrorGenerate("Computing training output vector");

    // Uncomment the below code to generate the tesing output data
    //
    // block_size = 1024;
    // grid_size = (testing_data_size + (block_size-1))/ block_size;

    // compute_output<<<grid_size, block_size >>>(dTestingOutput, dTestingData, input_dim, testing_data_size);
    // cudaDeviceSynchronize();
    // checkCUDAErrorGenerate("Computing testing output vector");
    //
    for(int d=0; d<input_dim; d++)
    {
        cudaMemcpy(input_data+(d*data_size), dInputData+(d*(data_size)), data_size*sizeof(double), cudaMemcpyDeviceToHost);
    }
    checkCUDAErrorGenerate("Memcpy");

    // Uncomment the below code to generate the tesing output data
    // 
    // for(int d=0; d<input_dim; d++)
    // {
    //     cudaMemcpy(testing_data+(d*testing_data_size), dTestingData+(d*(testing_data_size)), testing_data_size*sizeof(double), cudaMemcpyDeviceToHost);
    // }
    // checkCUDAErrorGenerate("Memcpy");
    // 

    cudaMemcpy(output_data, dOutputData, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToHost);
    checkCUDAErrorGenerate("Memcpy");

    // Uncomment the below code to generate the tesing output data
    // 
    // cudaMemcpy(testingOutput, dTestingOutput, testing_data_size*output_dim*sizeof(double), cudaMemcpyDeviceToHost);
    // checkCUDAErrorGenerate("Memcpy");
    // 

    curandDestroyGenerator(gen);

    //Free the memeory
    cudaFree(dInputData);
    cudaFree(dOutputData);
    cudaFree(dTestingData);
    cudaFree(dTestingOutput);
}
