#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include "local_input_system.cuh"

cudaEvent_t startLIS, stopLIS; 
float millisecondsLIS;


#define TIME_startLIS {cudaEventCreate(&startLIS); cudaEventCreate(&stopLIS); cudaEventRecord(startLIS);}
#define TIME_stopLIS(a) {cudaEventRecord(stopLIS); cudaEventSynchronize(stopLIS); cudaEventElapsedTime(&millisecondsLIS, startLIS, stopLIS); printf("%s: Elapsed time: %lf ms\n", a, millisecondsLIS); }


void compute_stencil_array(uint64_t* map_index, int stencil_size, int point_count, int boundry_point_size, int r, uint64_t* d_s_array=NULL)
{
	int current_idx;
	int current_idx_1;
	int current_idx_2;

	// Stencil array!
	if(stencil_size > 0)
	{
		uint64_t* s_array;
		s_array = (uint64_t*) malloc(r*sizeof(uint64_t));
		
		for(int i = -(r/2); i <= r/2; i++)
		{
			if(i==0){
				s_array[i] = 0;
			}else if( i < 0){
				current_idx = -1 * i;
				s_array[current_idx] = (uint64_t) i;
			}else{
				current_idx = i + (r/2);
				s_array[current_idx] = (uint64_t) i;
				}
		}
		cudaMemcpy(d_s_array, s_array, r*sizeof(uint64_t), cudaMemcpyHostToDevice);
		free(s_array);
	}
	
        
    	//Boundry array
	uint64_t* b1_array;
	b1_array = (uint64_t*) malloc(boundry_point_size*sizeof(uint64_t));
	for(int i = 0; i < boundry_point_size; i++)
	{
		current_idx_1 = i/r;
		current_idx_2 = i%r;
    	if(current_idx_1==0){
			b1_array[i] = (uint64_t)current_idx_2;
		}else{
			if(current_idx_2 == 0) b1_array[i] = (uint64_t) current_idx_1;
			if(current_idx_2 <= current_idx_1 && current_idx_2 != 0) b1_array[i] = (uint64_t)(current_idx_1 - current_idx_2);
			if(current_idx_2 > current_idx_1 && current_idx_2 != 0) b1_array[i] = (uint64_t)( current_idx_2);            
    	}
    }
	cudaMemcpy(map_index, b1_array, boundry_point_size*sizeof(uint64_t), cudaMemcpyHostToDevice);

    //Boundry array
	uint64_t* b2_array;
	b2_array = (uint64_t*) malloc(boundry_point_size*sizeof(uint64_t));
	int s_point = (point_count - r/2);
	int counter = r/2-1;
    
	for(int i = 0; i < boundry_point_size; i++)
	{
		current_idx_1 = i/r;
		current_idx_2 = i%r;
		b2_array[i] = 0;
		if(current_idx_1==(r/2 - 1)){
			b2_array[i] = (uint64_t) (s_point - current_idx_2);
		}else{
			if(current_idx_2 == 0) b2_array[i] = (uint64_t) s_point;
			if(current_idx_2 > 0 && current_idx_2 <= counter) b2_array[i] = (uint64_t)(s_point + current_idx_2);
			if(current_idx_2 > 0 && current_idx_2 > counter) b2_array[i] = (uint64_t)(s_point - r + current_idx_2);
		}
		if(current_idx_2 == (r-1)){
    		s_point++;
			counter--;
		} 
	}
	cudaMemcpy(map_index+boundry_point_size+stencil_size, b2_array, boundry_point_size*sizeof(uint64_t), cudaMemcpyHostToDevice);

	
	free(b1_array);
	free(b2_array);
}
    
__global__ void copy_index_map(uint64_t* map, uint64_t* global_index, int point_count, int r)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=(point_count*r)) return;
	map[idx] = (uint64_t) idx / r;
}
    
__global__ void copy_stencil_map(uint64_t* map, uint64_t* d_s_array, int size, int r)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=size) return;
	map[idx] = d_s_array[idx%r];
}

__global__ void fill_indices(uint64_t* indices, int count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx>=count) return;

	indices[idx] = (uint64_t)idx;

	return;
}
    
void compute_map_index(uint64_t* map_index, uint64_t* map_temp_1, uint64_t* map_temp_2, int stencil_size, int point_count, int r)
{
	int size = point_count*r;
	int boundry_point_size = (r/2) * r;
    
	thrust::device_ptr<uint64_t> d_map_index =  thrust::device_pointer_cast(map_index);
	thrust::device_ptr<uint64_t> d_temp_map_1 =  thrust::device_pointer_cast(map_temp_1);
	thrust::device_ptr<uint64_t> d_temp_map_2 =  thrust::device_pointer_cast(map_temp_2);
	thrust::transform(d_temp_map_1 + boundry_point_size, d_temp_map_1 + (size-boundry_point_size), d_temp_map_2,
						d_map_index + boundry_point_size, thrust::plus<uint64_t>());
	map_index = thrust::raw_pointer_cast(d_map_index);    
}
    
    
__global__ void compute_local_system_coordinate( double* coords_local, double* coords_sorted, uint64_t* map, int point_count, int r)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx>=(point_count*r)) return;

	uint64_t index = map[idx];
	coords_local[idx] = coords_sorted[index];
}
    
void compute_local_input_data(double* localInputData, double** sortedDataPoints, uint64_t* map_index, int point_count, int dim, int r)
{
	uint64_t* map_temp_1;
	cudaMalloc((void**)&map_temp_1, point_count*r*sizeof(uint64_t));
        
	uint64_t* global_index;
	cudaMalloc((void**)&global_index, point_count*sizeof(uint64_t));
    
	// calculate GPU thread configuration	
	int block_size = 512;
	int grid_size = (point_count + (block_size - 1)) / block_size;
    
	// generate index array initially set to 1:point_count	
	fill_indices<<<grid_size, block_size>>>(global_index, point_count);
	cudaDeviceSynchronize();
	// cudaMemcpy(h_array, global_index, point_count*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	grid_size = ((point_count*r) + (block_size - 1)) / block_size;
        
	// TIME_start
	copy_index_map<<<grid_size, block_size>>>(map_temp_1, global_index, point_count, r);
	cudaDeviceSynchronize();
	// cudaMemcpy(h_array, map_temp_2, point_count*r*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// TIME_stop("Copy indexes")
    
	int boundry_point_size = (r/2 * r);
	int stencil_size = (point_count*r) - (2*boundry_point_size);
        
	
	if(stencil_size > 0)
	{
		uint64_t* d_s_array;
		cudaMalloc((void**)&d_s_array, r*sizeof(uint64_t));
		
		uint64_t* map_temp_2;
		cudaMalloc((void**)&map_temp_2, (stencil_size)*sizeof(uint64_t));
		
		compute_stencil_array(map_index, stencil_size, point_count, boundry_point_size, r, d_s_array);

		grid_size = (stencil_size + (block_size - 1)) / block_size;

		copy_stencil_map<<<grid_size, block_size>>>(map_temp_2, d_s_array, stencil_size, r);
		cudaDeviceSynchronize();
		TIME_startLIS
		compute_map_index(map_index, map_temp_1, map_temp_2, stencil_size, point_count, r);
		TIME_stopLIS("Compute Map Index")
		cudaFree(map_temp_2);
		cudaFree(d_s_array);
	}else{
		TIME_startLIS
		compute_stencil_array(map_index, stencil_size, point_count, boundry_point_size, r);
		TIME_stopLIS("Compute Map Index")
	}
    
	TIME_startLIS
	for (int d=0; d<dim; d++)
	{
		compute_local_system_coordinate<<<grid_size, block_size>>>(localInputData+(d*(point_count*r)), sortedDataPoints[d], map_index, point_count, r);
	}
    TIME_stopLIS("Local input system computation")

    
	cudaFree(map_temp_1);
	cudaFree(global_index);
}
    