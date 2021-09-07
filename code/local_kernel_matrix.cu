#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<curand.h>
#include<device_launch_parameters.h>
#include "local_kernel_matrix.cuh"

#define BLOCK_SIZE_2D 32

#define SIGMA 2
const double lamda = 0.0000000001;

cudaEvent_t startLKM, stopLKM; 
float millisecondsLKM;


#define TIME_startLKM {cudaEventCreate(&startLKM); cudaEventCreate(&stopLKM); cudaEventRecord(startLKM);}
#define TIME_stopLKM(a) {cudaEventRecord(stopLKM); cudaEventSynchronize(stopLKM); cudaEventElapsedTime(&millisecondsLKM, startLKM, stopLKM); printf("%s: Elapsed time: %lf ms\n", a, millisecondsLKM); }


void checkCUDAErrorLKRR(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
}

int iDivUp(int hostPtr, int b)
{
	 return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}

__global__ void kernel_matrix_computation(double* kernel_matrix, double* coords_d, int point_count, int dim, int r)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if(tidx<r*r && tidy < point_count)
	{
		int idx_local = tidx/r; 
		int idy_local = tidx%r;
		double norm = 0;
		double kernel_value =0;
		for(int d=0; d<dim; d++)
		{
			norm += pow(coords_d[idx_local+(tidy*r)+(d*r*point_count)]-coords_d[idy_local+(tidy*r)+(d*r*point_count)],2);
		}
		norm = sqrt(norm);
		kernel_value = exp(-pow(norm,2)/(SIGMA));
		kernel_matrix[tidx+(tidy*r*r)] = kernel_value;
		// printf("----%i, %i, %i, %i, %i, %f, %f \n",tidx, tidy, idx_local, idy_local, tidx+(tidy*r*r), norm, kernel_value);
	}
}

__global__ void regularize_kernel_matrix(double* kernel_matrix, int point_count, int r, double lambda)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;

	if(tidx>=r*point_count) return;
	int idx_p = tidx / r;
	int idx_q = tidx % r;

	kernel_matrix[(idx_p*r*r)+(idx_q*r)+idx_q] += lambda;
}

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

//     for(int i = 0; i < dim; i++)
//     { 
//         temp_array[i] = pow((x_i[i]) - (x_j[i]), 2) ;
//         // printf("%f ", temp_array[i]);
//         l1_norm += temp_array[i];
//     }
// 	l1_norm = sqrt(l1_norm);
	
// 	free(x_i);
//     free(x_j);
//     return l1_norm;
// }


void compute_local_kernel_matrix(double* kernelMatrix, double* localInputData, int dim, int point_count, int r)
{

	// double* host_arr;
	// double* h_kernelMat;
	// host_arr = (double*) malloc(r*dim*sizeof(double));
	// h_kernelMat = (double*) malloc(r*r*sizeof(double));
	

	// double l1_norm_value; 
	// for(int k=0; k<point_count; k++)
	// {
	// 	for(int i=0; i<dim; i++){
	// 		cudaMemcpy(host_arr+(i*r), localInputData+((i*point_count*r)+(k*r)), r*sizeof(double), cudaMemcpyDeviceToHost);
	// 	}
	// 	for(int i = 0; i < r; i++)
	// 	{
	// 		for (int j = 0; j < r; j++)
	// 		{
	// 			l1_norm_value = l1_norm(host_arr, i, j, r, dim);
	// 			h_kernelMat[i*r+j] = exp(-(pow(l1_norm_value, 2)/SIGMA));
	// 		}
	// 	}
	// 	for(int i=0; i<r; i++){
	// 		for(int j=0; j<r; j++){
	// 			printf("%f ", h_kernelMat[i*r+j]);
	// 		}
	// 	}
	// 	printf("\n");
	// }   
   
	
	dim3 block_size(BLOCK_SIZE_2D ,BLOCK_SIZE_2D);
	dim3 grid_size(iDivUp(r*r, BLOCK_SIZE_2D) ,iDivUp(point_count, BLOCK_SIZE_2D));

	TIME_startLKM
	kernel_matrix_computation<<<grid_size, block_size>>>(kernelMatrix, localInputData, point_count, dim, r);
	cudaDeviceSynchronize();
	TIME_stopLKM("Local kernel matrix computation")
	checkCUDAErrorLKRR("Building the local kernel matrix");

	// grid_size(iDivUp(r, BLOCK_SIZE_2D) ,iDivUp(point_count, BLOCK_SIZE_2D));

	int blockSize = 1024;
	int gridSize = ((point_count*r) + (blockSize-1))/blockSize;
	
	// TIME_startLKM
	regularize_kernel_matrix<<<gridSize, blockSize>>> (kernelMatrix, point_count, r, lamda);
	cudaDeviceSynchronize();
	// TIME_stopLKM("Regularize kernel matrices")
	checkCUDAErrorLKRR("Regularizing the local kernel matrix");

	
	
}
