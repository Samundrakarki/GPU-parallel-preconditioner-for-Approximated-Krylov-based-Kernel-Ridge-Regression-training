
// Copyright (C) 2016 Peter Zaspel
//
// This file is part of hmglib.
//
// hmglib is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// hmglib is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with hmglib.  If not, see <http://www.gnu.org/licenses/>.
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include "nearest_neighbors.cuh"

cudaEvent_t startNN, stopNN; 
float millisecondsNN;


#define TIME_startNN {cudaEventCreate(&startNN); cudaEventCreate(&stopNN); cudaEventRecord(startNN);}
#define TIME_stopNN(a) {cudaEventRecord(stopNN); cudaEventSynchronize(stopNN); cudaEventElapsedTime(&millisecondsNN, startNN, stopNN); printf("%s: Elapsed time: %lf ms\n", a, millisecondsNN); }

void checkCUDAErrorNN(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void print_binary(uint64_t val)
{
	char c;
	for (int i=0; i<64; i++)
	{
		if ((val & 0x8000000000000000u)>0)
			c='1';
		else
			c='0';
		val = val << 1;
		printf("%c",c);
		
	}
	printf("\n");
}



__global__ void fill_with_indices(uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx>=count) return;

	indices[idx] = (uint64_t)idx;

	return;
}

__global__ void reorder_by_index(double* output, double* input, uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=count) return;

	uint64_t ind = indices[idx];

	output[idx] = input[ind];

	return;
}

__global__ void init_point_set(struct point_set* points, double** coords_device, int dim, double* max_per_dim, double* min_per_dim, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=1) return;

	points->dim = dim;
	points->size = size;
	points->coords = coords_device;
	points->max_per_dim = max_per_dim;
	points->min_per_dim = min_per_dim;

	return;
}

__global__ void init_morton_code(struct morton_code* morton, uint64_t* code, int dim, int bits, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx>=1) return;

	morton->code = code;
	morton->dim = dim;
	morton->bits = bits;
	morton->size = size;
}


void get_morton_ordering(struct point_set* points_d, struct morton_code* morton_d, uint64_t* order)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	struct morton_code morton_h;
	cudaMemcpy(&morton_h, morton_d, sizeof(struct morton_code), cudaMemcpyDeviceToHost);
	
	int point_count = points_h.size;

	thrust::device_ptr<uint64_t> morton_codes_ptr = thrust::device_pointer_cast(morton_h.code);
	thrust::device_ptr<uint64_t> order_ptr = thrust::device_pointer_cast(order);

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;

	
	// generate index array initially set to 1:point_count	
	fill_with_indices<<<grid_size, block_size>>>(order, point_count);

	// find ordering of points following Z curve
	// sort_by_key works as follows: First it sorts the morton code and then following the morton code it will sort the order_pts
	//						Unsorted morton code:	{(first_morton, 1), (second_morton, 2), ....., (last_morton, point_count)}
	//						Sorted morton code : {(sorted_first_morton, 1), (sorted_second_morton, idx_unsorted), ....., (sorted_last_morton, idx_unsorted)}
	// 						idx_unsorted: index where the morton code was located before sorting 
	TIME_startNN;
	thrust::sort_by_key(morton_codes_ptr, morton_codes_ptr + point_count, order_ptr);
	TIME_stopNN("Sorting the index using morton code");
}

void print_points(struct point_set* points_d)
{	
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}


	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			printf("%lf ", coords_h[d][p]);
		}
		printf("\n");
	}
	
	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;

}

void write_points(struct point_set* points_d, char* file_name)
{	
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}

	FILE* f = fopen(file_name,"w");	

	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			fprintf(f,"%lf ", coords_h[d][p]);
		}
		fprintf(f,"\n");
	}
	
	fclose(f);

	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;

}


void reorder_point_set(struct point_set* points_d, uint64_t* order)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);

	double* coords_tmp;
	cudaMalloc((void**)&coords_tmp, point_count*sizeof(double));
	checkCUDAErrorNN("NN: cudaMalloc 1");

	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;

	TIME_startNN
	for (int d=0; d<dim; d++)
	{
		reorder_by_index<<<grid_size, block_size>>>(coords_tmp, coords_d_host[d], order, point_count);
		cudaMemcpy(coords_d_host[d], coords_tmp, point_count*sizeof(double), cudaMemcpyDeviceToDevice);	
	}
	TIME_stopNN("Reorder the data points");
	
	cudaFree(coords_tmp);
	delete[] coords_d_host;
}


void print_points_with_morton_codes(struct point_set* points_d, struct morton_code* code_d)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;
	int point_count = points_h.size;
	
	double** coords_d_host = new double*[dim];
	cudaMemcpy(coords_d_host, points_h.coords, sizeof(double*)*dim, cudaMemcpyDeviceToHost);
	
	double** coords_h = new double*[dim];
	for (int d=0; d<dim; d++)
	{
		coords_h[d] = new double[point_count];
		cudaMemcpy(coords_h[d], coords_d_host[d], sizeof(double)*point_count, cudaMemcpyDeviceToHost);
	}

	struct morton_code code_h;
	cudaMemcpy(&code_h, code_d, sizeof(struct morton_code), cudaMemcpyDeviceToHost);

	uint64_t* codes_h = new uint64_t[point_count];
	cudaMemcpy(codes_h, code_h.code, point_count*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	
	for (int p=0; p<point_count; p++)
	{
		for (int d=0; d<dim; d++)
		{
			printf("%lf ", coords_h[d][p]);
		}
		printf("\n");
		print_binary(codes_h[p]);
	}
	
	for (int d=0; d<dim; d++)
		delete [] coords_h[d];
	delete [] coords_h;
	delete [] coords_d_host;	
}

void get_min_and_max(double* min, double* max, double* values, int size)
{
	thrust::device_ptr<double> values_ptr =  thrust::device_pointer_cast(values);
	thrust::pair<thrust::device_ptr<double>,thrust::device_ptr<double> > minmax = thrust::minmax_element(values_ptr, values_ptr + size);
	*min = *minmax.first;
	*max = *minmax.second;
}

void compute_minmax(struct point_set* points_d)
{
	struct point_set points_h;
	cudaMemcpy(&points_h, points_d, sizeof(struct point_set), cudaMemcpyDeviceToHost);
	int dim = points_h.dim;

	double** coords_d = new double*[dim];
	cudaMemcpy(coords_d, points_h.coords, dim*sizeof(double*), cudaMemcpyDeviceToHost);

	// compute extremal values for the point set
	double* min_per_dim_h = new double[dim];
	double* max_per_dim_h = new double[dim];
	for (int d=0; d<dim; d++)
		get_min_and_max(&(min_per_dim_h[d]), &(max_per_dim_h[d]), coords_d[d], points_h.size);
	cudaMemcpy(points_h.max_per_dim, max_per_dim_h, points_h.dim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(points_h.min_per_dim, min_per_dim_h, points_h.dim*sizeof(double), cudaMemcpyHostToDevice);

	delete [] min_per_dim_h;
	delete [] max_per_dim_h;
	delete [] coords_d;
}

void compute_nearest_neighbors(double* inputData, double* outputData, double** sortedDataPoints, double* sorted_inData, 
									double* sorted_outData, uint64_t* order, int point_count, int dim, int bits, int r)
{      
	
// 	// allocating memory for point_count coordinates in dim dimensions
	double** coords_d;
	coords_d = new double*[dim];
	for (int d = 0; d < dim; d++)
	{
		cudaMalloc((void**)&(coords_d[d]), point_count*sizeof(double));
		checkCUDAErrorNN("NN: cudaMalloc 2.");
	}


// 	// allocating memory for extremal values per dimension
	double* max_per_dim_d;
	cudaMalloc((void**)&max_per_dim_d, dim*sizeof(double));
	checkCUDAErrorNN("NN: cudaMalloc 3");

	double* min_per_dim_d;
	cudaMalloc((void**)&min_per_dim_d, dim*sizeof(double));
	checkCUDAErrorNN("NN: cudaMalloc 4");


// 	// generating device pointer that holds the dimension-wise access
	double** coords_device;
	cudaMalloc((void**)&(coords_device), dim*sizeof(double*));
	checkCUDAErrorNN("NN: cudaMalloc 5");

	cudaMemcpy(coords_device, coords_d, dim*sizeof(double*), cudaMemcpyHostToDevice);
	
// 	// allocationg memory for morton codes
	uint64_t* code_d;
	cudaMalloc((void**)&code_d, point_count*sizeof(uint64_t));
	checkCUDAErrorNN("NN: cudaMalloc 6");


// 	// setting up data strcture for point set
	struct point_set* points_d;
	cudaMalloc((void**)&points_d, sizeof(struct point_set));
	checkCUDAErrorNN("NN: cudaMalloc 7");

	init_point_set<<<1,1>>>(points_d, coords_device, dim, max_per_dim_d, min_per_dim_d, point_count);

// 	// setting up data structure for morton code
	struct morton_code* morton_d;
	cudaMalloc((void**)&morton_d, sizeof(struct morton_code));
	checkCUDAErrorNN("NN: cudaMalloc 8");

	init_morton_code<<<1,1>>>(morton_d, code_d, dim, bits, point_count);

	//Setting up coords with the provided input data 
	for(int d = 0; d < dim; d++)
	{
		cudaMemcpy(coords_d[d], inputData+(d*point_count), point_count*sizeof(double), cudaMemcpyHostToDevice);
	}

//	// Writing unsoted points to a file
	char file_name[2000];
	sprintf(file_name,"unsorted_points.dat");
	write_points(points_d,file_name);

 	// compute extremal values for the point set
	compute_minmax(points_d);
	
 	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;

 	// generate morton codes
	TIME_startNN;
	get_morton_code(points_d, morton_d,grid_size, block_size);
	TIME_stopNN("Calculating the Morton code.");
	checkCUDAErrorNN("get_morton_code");

	// print_points_with_morton_codes(points_d, morton_d);
	get_morton_ordering(points_d, morton_d, order);

 	// reorder points following the morton code order
	reorder_point_set(points_d, order);	

	//reorder output data points
	block_size = 512;
	grid_size = (point_count + (block_size - 1)) / block_size;

	reorder_by_index<<<grid_size, block_size>>>(sorted_outData, outputData, order, point_count);
	cudaDeviceSynchronize();
	
	for(int d=0; d<dim; d++)
	{
		cudaMemcpy(sortedDataPoints[d], coords_d[d], point_count*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(sorted_inData+(d*point_count), sortedDataPoints[d], point_count*sizeof(double), cudaMemcpyDeviceToHost);
	}

	// print ordered points
	// print_points(points_d);

//	// write points to file
	sprintf(file_name,"sorted_points.dat");
	write_points(points_d,file_name);

// 	// freeing memory for morton codes
	cudaFree(code_d);
// 	// freeing coordinates memory
	for (int d = 0; d < dim; d++)
	{
		cudaFree(coords_d[d]);
	}
	cudaFree(points_d);
	cudaFree(morton_d);
	cudaFree(max_per_dim_d);
	cudaFree(min_per_dim_d);
	cudaFree(coords_device);
	delete [] coords_d;
	checkCUDAErrorNN("NN: Freeing Memory.");
}

// int main()
// {
//     double * input_data;
//     double* preconditioner;
    
//     int data_size = 500000;
// 	int input_dim = 3;
// 	int bits = 20;
// 	int r = 5;

//     input_data = (double* ) malloc(data_size * input_dim*sizeof(double));
//     preconditioner = (double*) malloc(data_size* data_size*sizeof(double));

//     double temp = 54.2;
//     for(int i = 0; i<input_dim; i++)
//     {
//         for(int j = 0; j<data_size; j++)
//         {
//             if((i*input_dim+j) % 2 == 0){
//                 input_data[j*input_dim+i] =  (double)(i*input_dim+j) + temp;
//             }else{
//                 input_data[j*input_dim+i] =  (double)(i*input_dim+j) - temp;
//             }
//         } 
//     }

//     // print_data(input_data, data_size, input_dim);

//     compute_nearest_neighbors(data_size, input_dim, bits, r);
	

//     free(input_data);
//     free(preconditioner);
// }