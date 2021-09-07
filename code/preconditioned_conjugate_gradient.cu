#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "preconditioned_conjugate_gradient.cuh"

cudaEvent_t start_cg, stop_cg; 
float milliseconds_cg;


#define TIME_start_CG {cudaEventCreate(&start_cg); cudaEventCreate(&stop_cg); cudaEventRecord(start_cg);}
#define TIME_stop_CG(a) {cudaEventRecord(stop_cg); cudaEventSynchronize(stop_cg); cudaEventElapsedTime(&milliseconds_cg, start_cg, stop_cg); printf("%s: Elapsed time: %lf ms\n", a, milliseconds_cg); }


//Function to check for error
int error_check(cublasStatus_t stat)
{
    if(stat == CUBLAS_STATUS_SUCCESS)
    {
        return 0;
    }else{
        if(stat == CUBLAS_STATUS_INVALID_VALUE){
            printf ("Error: CUBLAS_STATUS_INVALID_VALUE %d ", stat);
        }else if(stat == CUBLAS_STATUS_NOT_INITIALIZED){
            printf ("Error: CUBLAS_STATUS_NOT_INITIALIZED %d ", stat);
        }else if(stat == CUBLAS_STATUS_ALLOC_FAILED){
            printf ("Error: CUBLAS_STATUS_ALLOC_FAILED %d ", stat);
        }else if(stat == CUBLAS_STATUS_ARCH_MISMATCH){
            printf ("Error: CUBLAS_STATUS_ARCH_MISMATCH %d ", stat);
        }else if(stat == CUBLAS_STATUS_MAPPING_ERROR){
            printf ("Error: CUBLAS_STATUS_MAPPING_ERROR %d ", stat);
        }else if(stat == CUBLAS_STATUS_EXECUTION_FAILED){
            printf ("Error: CUBLAS_STATUS_EXECUTION_FAILED %d ", stat);
        }else if(stat == CUBLAS_STATUS_INTERNAL_ERROR){
            printf ("Error: CUBLAS_STATUS_INTERNAL_ERROR %d ", stat);
        }else{
            printf ("Error: Unknown Error %d ", stat);
        }
        return 1;
    }
}

void checkCUDAErrorCG(const char* msg) {
    cudaError_t err = cudaGetLastError();
      if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    }

void print_(double* vector, int dim, int size)
{
    for(int i = 0; i<dim; i++)
    {
        for(int j = 0; j<size; j++){
                printf("%f ", vector[i*size+j]);
        }
        printf("\n \n");
    }
}

//Function to compute the alpha and beta values when the dimension is greater than 1
__global__ void alpha_values_computation(double* alpha_matrix, double* inner_prod_1, double* inner_prod_2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        alpha_matrix[idx*size+idx] = inner_prod_1[idx*size+idx] / inner_prod_2[idx*size+idx];
    }
}

//Function to print the resulting matrix
void get_resulting_matrix(cublasStatus_t stat, int datasize, int dim, double* device_vector, double* vector)
{
    stat = cublasGetMatrix (datasize , dim , sizeof(*device_vector) , device_vector, datasize, vector, datasize);
    if(error_check(stat)){
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    print_(vector, dim, datasize);
}


//Resulting matrix is already in column major format. 
//Function to compute the inital values
void compute_initial_values(cublasHandle_t handle, double* device_kernel_matrix, double* device_approximated_solution, double* device_residual_prev, double* device_output_data, 
                            double* device_preconditioner, double* device_z_prev,double* device_search_dir,
                            double* residual, double* z, double* search_dir,
                            int data_size, int output_dim, const double* alpha, const double* beta, const double* beta_1)
{
    cublasStatus_t stat;

    //Compute r_0 = b - k x_0
    // k*x_0 = c
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, data_size,
                        alpha, device_kernel_matrix, data_size, device_approximated_solution, data_size,
                        beta, device_residual_prev, data_size);
    if(error_check(stat)){
        cudaDeviceReset();
        exit(EXIT_FAILURE);
     }
    // printf("Initial Values --------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_residual_prev, residual);

    // out_data - (A * x_0) 
    stat = cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, 
                        alpha, device_output_data, data_size, 
                        beta_1, device_residual_prev, data_size, device_residual_prev, data_size);
    if(error_check(stat)){
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // printf("R_0 --------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_residual_prev, residual);

    //Comput z_0 = P * r_0
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, data_size, 
                        alpha, device_preconditioner, data_size, device_residual_prev, data_size, 
                        beta, device_z_prev, data_size);
    if(error_check(stat)){
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // printf("Z_0 --------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_z_prev, z);
    
    //Compute d_0 = z_0
    cudaMemcpy(device_search_dir, device_z_prev, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCUDAErrorCG("Memcpy.");
    // printf("D_0 --------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_search_dir, search_dir);

}

//Function to compute the alpha values
void compute_alpha(cublasHandle_t handle, double* device_kernel_matrix, double* device_search_dir,double* device_temp_array, double* device_residual_prev,
                    double* device_z_prev, double* device_temp_array_2, double* device_temp_array_1, double* device_alpha_values,
                    double* temp_array, double* alpha_values, double* temp_array_1,
                    int data_size, int output_dim, const double* alpha, const double* beta)
{
    
    cublasStatus_t stat;

    // k * d_i = temp_array
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, data_size, 
                        alpha, device_kernel_matrix, data_size, device_search_dir, data_size, 
                        beta, device_temp_array, data_size);
    if(error_check(stat)){
        printf("Error in alpha computation 1.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // get_resulting_matrix(stat, data_size, output_dim, device_temp_array, temp_array);
    
    //r_i * z_i = device_temp_array_2
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, output_dim, data_size, 
                        alpha, device_residual_prev, data_size, device_z_prev, data_size, 
                        beta, device_temp_array_2, output_dim);
    if(error_check(stat)){
        printf("Error in alpha computation 2.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // get_resulting_matrix(stat, output_dim, output_dim, device_temp_array_2, temp_array_1);
    
    // d_i * temp_array
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, output_dim, data_size, 
                        alpha, device_search_dir, data_size, device_temp_array, data_size, 
                        beta, device_temp_array_1, output_dim);
    if(error_check(stat)){
        printf("Error in alpha computation 3.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // get_resulting_matrix(stat, output_dim, output_dim, device_temp_array_1, temp_array_1);

    if(output_dim == 1)
    {
        alpha_values_computation<<<1, 1>>>(device_alpha_values, device_temp_array_2, device_temp_array_1, output_dim);
        cudaDeviceSynchronize();
        // printf("Alpha Values------\n");
        // get_resulting_matrix(stat, output_dim, output_dim, device_alpha_values, alpha_values);  
    }
    else if(output_dim > 1)
    {
        int block_size = 1024;
        int grid_size = output_dim*output_dim + (block_size - 1) / block_size;

        alpha_values_computation<<<grid_size, block_size>>>(device_alpha_values, device_temp_array_2, device_temp_array_1, output_dim);
        cudaDeviceSynchronize();
        // printf("Alpha Values------\n");
        // get_resulting_matrix(stat, output_dim, output_dim, device_alpha_values, alpha_values);
    }
}

//Function to compute the new approximated solution 
void compute_new_approximated_solution(cublasHandle_t handle, double* device_search_dir, double* device_alpha_values, 
                                        double* device_approximated_solution, double* approximated_solution,
                                        int data_size, int output_dim, const double* alpha, const double* beta)
{   
    cublasStatus_t stat;
    // x_i+1 = x_i + alpha_i * d_i
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, output_dim, 
                        alpha, device_search_dir, data_size, device_alpha_values, output_dim, 
                        beta, device_approximated_solution, data_size);
    if(error_check(stat)){
        printf("Error in new approximation computation.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    // printf("Approximated Solution ----------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_approximated_solution, approximated_solution);
}

//Function to compute the new residual
void compute_new_residual(cublasHandle_t handle, double* device_temp_array, double* device_alpha_values, 
                            double* device_residual_next, double* residual,
                            int data_size, int output_dim, const double* alpha, const double* beta)
{
    cublasStatus_t stat;

    //temp_array(k * d_i)
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, output_dim, 
                        alpha, device_temp_array, data_size, device_alpha_values, output_dim, 
                        beta, device_residual_next, data_size);
    if(error_check(stat)){
        printf("Error in new residual computation.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // printf("R_next----------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_residual_next, residual);    
}

//Function to compute the new z 
void compute_new_z(cublasHandle_t handle, double* device_preconditioner, double* device_residual_next, double* device_z_next, double* z, 
                    int data_size, int output_dim, const double* alpha, const double* beta)
{    
    cublasStatus_t stat;

    //Comput z_i+1 = P * r_i+1
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, data_size, 
                        alpha, device_preconditioner, data_size, device_residual_next, data_size, 
                        beta, device_z_next, data_size);
    if(error_check(stat)){
        printf("Error in new z computation.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    // printf("Z_next----------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_z_next, z);
}

//Function to compute the beta
void compute_beta(cublasHandle_t handle, double* device_residual_next, double*  device_z_next, double* device_temp_array_1, double* device_temp_array_2, 
                    double* device_beta_values, double* beta_values, double* temp_array_1,  int data_size, int output_dim, const double* alpha, const double* beta)
{
    cublasStatus_t stat;

    // r_i * z_i = device_temp_array_2
    // r_i+1 * z_i+1 = device_temp_array_1
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, output_dim, data_size, 
                        alpha, device_residual_next, data_size, device_z_next, data_size, 
                        beta, device_temp_array_1, output_dim);
    if(error_check(stat)){
        printf("Error in beta computation.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // get_resulting_matrix(stat, output_dim, output_dim, device_temp_array_1, temp_array_1);
    if(output_dim == 1)
    {
        alpha_values_computation<<<1, 1>>>(device_beta_values, device_temp_array_1, device_temp_array_2, output_dim);
        // printf("Beta Values---------------\n");
        // get_resulting_matrix(stat, output_dim, output_dim, device_beta_values, beta_values);
        
    }else if(output_dim > 1){
        int block_size = 1024;
        int grid_size = output_dim*output_dim + (block_size - 1) / block_size;

        alpha_values_computation<<<grid_size, block_size>>>(device_beta_values, device_temp_array_1, device_temp_array_2, output_dim);
        cudaDeviceSynchronize();
        // get_resulting_matrix(output_dim, output_dim, device_beta_values, beta_values);
    }else{
        printf("Error in dimension of output");
    }
}

//Function to compute new search direction
void compute_new_search_dir(cublasHandle_t handle, double* device_search_dir_next, double* device_z_next, double* device_search_dir,
                            double* device_beta_values, double* search_dir,  
                            int data_size, int output_dim, const double* alpha, const double* beta)
{
    cublasStatus_t stat;

    // d_i+1 = z_i+1
    cudaMemcpy(device_search_dir_next, device_z_next, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCUDAErrorCG("Memcpy.");
    
    // get_resulting_matrix(stat, data_size, output_dim, device_search_dir_next, search_dir);
    
    // d_i+1 = z_i+1 + d_i * beta_values
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, data_size, output_dim, output_dim, 
                        alpha, device_search_dir, data_size, device_beta_values, output_dim, 
                        beta, device_search_dir_next, data_size);
    if(error_check(stat)){
        printf("Error in new search direction computation.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // printf("New Search Direction -------\n");
    // get_resulting_matrix(stat, data_size, output_dim, device_search_dir_next, search_dir);
}

//Function to compute the error 
void compute_error(cublasHandle_t handle, double* device_residual_prev, double* device_search_dir, double* device_error,
                    double* error, int data_size,int output_dim, const double* alpha, const double* beta)
{
    cublasStatus_t stat;

    // error = r_i * r_i 
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, output_dim, data_size, alpha, device_residual_prev, data_size, device_residual_prev, data_size, beta, device_error, output_dim);
    if(error_check(stat)){
        printf("Error in computing error-1.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    stat = cublasGetMatrix (output_dim , output_dim , sizeof(*device_error) , device_error, output_dim, error, output_dim);
    if(error_check(stat)){
        printf("Error in compting error-2.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    if(output_dim > 1)
    {
        for(int i =1; i < output_dim; i++)
        {
            error[0] +=  error[i*output_dim+i];
        }
        error[0] = error[0]/output_dim;
        // printf("%f ", error[0]);
    }
    // error[0] = sqrt(error[0]);
    // printf("Error -------\n");
    // printf("%f \n", error[0]);
}


void preconditioned_conjugate_gradient(double* kernel_matrix, double* preconditioner, double* output_data, 
                                        double* approximated_solution, int data_size, int output_dim)

{
    
    //Host arrays for cg method
    double* residual;
    double* z;
    double* search_dir;
    double* alpha_values;
    double* beta_values;
    double* error;
    double* temp_array;
    double* temp_array_1;

    //initialize vectors and matrix needed for device(GPU)
    double* device_kernel_matrix;
    double* device_output_data;
    double* device_approximated_solution;
    double* device_residual_prev;
    double* device_residual_next;
    double* device_z_prev;
    double* device_z_next;
    double* device_search_dir;
    double* device_search_dir_next;
    double* device_preconditioner;
    double* device_alpha_values;
    double* device_beta_values;
    double* device_error;
    double* device_temp_array;
    double* device_temp_array_1;
    double* device_temp_array_2;

    //CUBLAS function status --stat
    cublasStatus_t stat;
    //CUBLAS context -- handle
    cublasHandle_t handle;
    
     //Create context
     stat = cublasCreate(&handle);

//Host memeory allocation
    residual = (double*) calloc(data_size*output_dim, sizeof(double));
    z = (double*) calloc(data_size*output_dim, sizeof(double));
    search_dir = (double*) calloc(data_size*output_dim, sizeof(double));  
    alpha_values = (double*) calloc(output_dim*output_dim, sizeof(double));
    beta_values = (double*) calloc(output_dim*output_dim, sizeof(double));
    error = (double*) calloc(output_dim*output_dim, sizeof(double));
    temp_array = (double*) calloc(data_size*output_dim, sizeof(double));
    temp_array_1 = (double*) calloc(output_dim*output_dim, sizeof(double));


    //Device memory allocation
    cudaMalloc((void**)&device_kernel_matrix, data_size * data_size * sizeof(double));
    checkCUDAErrorCG("cudaMallocCG: 1.");
    cudaMalloc((void**)&device_preconditioner, data_size*data_size* sizeof(*preconditioner));
    checkCUDAErrorCG("cudaMallocCG: 2.");
    cudaMalloc((void**)&device_output_data, data_size * output_dim * sizeof(*output_data)); 
    checkCUDAErrorCG("cudaMallocCG: 3.");

    cudaMalloc((void**)&device_approximated_solution, data_size*output_dim* sizeof(*approximated_solution));
    checkCUDAErrorCG("cudaMallocCG: 4.");
    cudaMalloc((void**)&device_residual_prev, data_size* output_dim * sizeof(*residual));
    checkCUDAErrorCG("cudaMallocCG: 5.");
    cudaMalloc((void**)&device_residual_next, data_size* output_dim * sizeof(*residual));
    checkCUDAErrorCG("cudaMallocCG: 6.");
    cudaMalloc((void**)&device_z_prev, data_size* output_dim * sizeof(*z));
    checkCUDAErrorCG("cudaMallocCG: 7.");
    cudaMalloc((void**)&device_z_next, data_size * output_dim * sizeof(*z));
    checkCUDAErrorCG("cudaMallocCG: 8.");

    cudaMalloc((void**)&device_error, output_dim * output_dim * sizeof(*error));
    checkCUDAErrorCG("cudaMallocCG: 9.");
    cudaMalloc((void**)&device_search_dir, data_size * output_dim * sizeof(*search_dir));
    checkCUDAErrorCG("cudaMallocCG: 10.");
    cudaMalloc((void**)&device_search_dir_next, data_size * output_dim * sizeof(*search_dir));
    checkCUDAErrorCG("cudaMallocCG: 11.");
    
    cudaMalloc((void**)&device_alpha_values, output_dim * output_dim * sizeof(*alpha_values));
    checkCUDAErrorCG("cudaMallocCG: 12.");
    cudaMalloc((void**)&device_beta_values, output_dim * output_dim * sizeof(*beta_values));
    checkCUDAErrorCG("cudaMallocCG: 13.");
    
    cudaMalloc((void**)&device_temp_array, data_size * output_dim * sizeof(*temp_array));
    checkCUDAErrorCG("cudaMallocCG: 14.");
    cudaMalloc((void**)&device_temp_array_1, output_dim * output_dim * sizeof(*temp_array_1));
    checkCUDAErrorCG("cudaMallocCG: 15.");
    cudaMalloc((void**)&device_temp_array_2, output_dim * output_dim * sizeof(*temp_array_1));
    checkCUDAErrorCG("cudaMallocCG: 16.");

   
   
    //Copying the data to device using cublasSetMatrix 
    // TIME_start_CG
    stat = cublasSetMatrix(data_size, data_size, sizeof(*kernel_matrix), kernel_matrix, data_size, device_kernel_matrix, data_size);
     if(error_check(stat)){
        printf("Error in Setting Matrix 1.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    // TIME_stop_CG("Setting matrix")

    stat = cublasSetMatrix(data_size, data_size, sizeof(*preconditioner), preconditioner, data_size, device_preconditioner, data_size);
    if(error_check(stat)){
        printf("Error in Setting Matrix 2.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    if(output_dim > 0){
        // cudaStat = cudaMemcpy(device_preconditioner, preconditioner, data_size*data_size*sizeof(double),cudaMemcpyHostToDevice);
        
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*output_data), output_data, data_size, device_output_data, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 3.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*residual), residual, data_size, device_residual_prev, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 4.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);;
        }
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*approximated_solution), approximated_solution, data_size, device_approximated_solution, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 5.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*z), z, data_size, device_z_prev, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 6.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*z), z, data_size, device_z_next, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 7.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(data_size, output_dim, sizeof(*search_dir), search_dir, data_size, device_search_dir, data_size);
        if(error_check(stat)){
            printf("Error in Setting Matrix 8.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(output_dim, output_dim, sizeof(*alpha_values), alpha_values, output_dim, device_alpha_values, output_dim);
        if(error_check(stat)){
            printf("Error in Setting Matrix 9.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix(output_dim, output_dim, sizeof(*error), error, output_dim, device_error, output_dim);
        if(error_check(stat)){
            printf("Error in Setting Matrix 10.\n");
            cudaDeviceReset();
            exit(EXIT_FAILURE);
        }
    }else{
        printf("Error in the dimension of the output vector");
        return;
    }


    int current_itr = 0;
    double tol;
    double al = 1.0f;
    double bet = 0.0f;
    double bet_1 = -1.0f;

    const double* alpha = &al;
    const double* beta = &bet;
    const double* beta_1 = &bet_1;


    
    //Compute r_0, z_0 and d_0
    compute_initial_values(handle, device_kernel_matrix, device_approximated_solution, device_residual_prev,
                            device_output_data, device_preconditioner, device_z_prev, device_search_dir, 
                            residual, z, search_dir,
                            data_size, output_dim, alpha, beta, beta_1);
    cudaMemcpy(device_residual_next, device_residual_prev, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
    checkCUDAErrorCG("Memcpy.");

    compute_error( handle, device_residual_prev, device_search_dir, device_error, error, 
                    data_size, output_dim, alpha, beta);
    
    tol = error[0] * pow(10, -8);
    // printf("%f \n", tol);

    TIME_start_CG
    while(current_itr < (data_size*100) && error[0] > tol)
    {      
        compute_alpha(handle, device_kernel_matrix,  device_search_dir, device_temp_array, device_residual_prev,
            device_z_prev, device_temp_array_2, device_temp_array_1, device_alpha_values,
            temp_array, alpha_values, temp_array_1,
            data_size, output_dim, alpha, beta);
            
        compute_new_approximated_solution(handle, device_search_dir, device_alpha_values, 
            device_approximated_solution, approximated_solution, data_size, output_dim, alpha, alpha);  

        compute_new_residual(handle, device_temp_array, device_alpha_values, device_residual_next, residual,
            data_size, output_dim, beta_1, alpha);

        compute_new_z(handle, device_preconditioner, device_residual_next, device_z_next, z, 
            data_size, output_dim, alpha, beta);
        
        compute_beta(handle, device_residual_next,  device_z_next, device_temp_array_1, device_temp_array_2, 
                 device_beta_values, beta_values, temp_array_1,  data_size, output_dim, alpha, beta);
        
        compute_new_search_dir( handle, device_search_dir_next, device_z_next, device_search_dir,
                    device_beta_values, search_dir,  
                    data_size, output_dim, alpha, alpha);
        
        //Copy the r_1 to r_0, z_1 to z_0 and d_1 to d_0
        cudaMemcpy(device_residual_prev, device_residual_next, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
        checkCUDAErrorCG("Memcpy.");
        
        cudaMemcpy(device_z_prev, device_z_next, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
        checkCUDAErrorCG("Memcpy.");
    
        cudaMemcpy(device_search_dir, device_search_dir_next, data_size*output_dim*sizeof(double), cudaMemcpyDeviceToDevice);
        checkCUDAErrorCG("Memcpy.");
        
        compute_error( handle, device_residual_prev, device_search_dir, device_error, error, 
            data_size, output_dim, alpha, beta);

        current_itr++;
    }
    TIME_stop_CG("Total time taken by CG method")
    
    printf("\nThe number of iteration is %i\n", current_itr);

    stat = cublasGetMatrix (data_size , output_dim , sizeof(*device_approximated_solution) , device_approximated_solution, data_size, approximated_solution, data_size);
    if(error_check(stat)){
        printf("Error in Getting  approixmated solution.\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    
    free(residual);
    free(z);
    free(search_dir);
    free(alpha_values);
    free(beta_values);
    free(error);
    free(temp_array);
    free(temp_array_1);

    cudaFree(device_kernel_matrix);
    cudaFree(device_output_data);
    cudaFree(device_approximated_solution);
    cudaFree(device_residual_prev);
    cudaFree(device_residual_next);
    cudaFree(device_z_prev);
    cudaFree(device_z_next);
    cudaFree(device_search_dir);
    cudaFree(device_search_dir_next);
    cudaFree(device_preconditioner);
    cudaFree(device_alpha_values);
    cudaFree(device_beta_values);
    cudaFree(device_error);
    cudaFree(device_temp_array);
    cudaFree(device_temp_array_1);
    cudaFree(device_temp_array_2);

    cublasDestroy(handle);
}

/**
Uncomment the section below and test if the cg method works for a hilbert matrix. to run type following command in terminal:
nvcc -lcublas filename.cu 
*/


// int main()
// {
//     int size = 3000;
//     int dim = 1;
    
//     double* kernel_matrix;
//     double* output_data;
//     double* preconditioner;
//     double* approximated_solution;

//     kernel_matrix = (double* ) malloc(size*size*sizeof(double));
//     preconditioner = (double* ) calloc(size*size, sizeof(double));
//     output_data = (double* ) malloc(size*dim*sizeof(double));
//     approximated_solution = (double*) calloc(size*dim, sizeof(double));
    
//     // kernel_matrix =  {1, 1/2, 1/3, 1/4, 
//     //                   1/2, 1/3, 1/4, 1/5,
//     //                   1/3, 1/4, 1/5, 1/6,
//     //                   1/4, 1/5, 1/6, 1/7};
//     double temp = 1;
//     for(int i = 0; i < size; i++){
//         for(int j = 0; j < size; j++){
//             // printf("%f ", 1/(temp+j));
//             kernel_matrix[i*size+j]=  1/(temp + j);
//         }
//         temp += 1;
//     }
//     // print_(kernel_matrix, 4, 4);

//     //Be carefull on setting the matrix; cublas works on column major format. So, store the matrix in column major format.
//     temp = 4;
//     double temp1 = 1;
//     for(int i = 0; i < dim; i++)
//     {
//         for(int j = 0; j < size; j++){
//             if(i==0){
//                 output_data[i*size+j] = 1/(temp+j);
//             }else{
//                 output_data[i*size+j] = 1/(temp1+j);
//             }
//         }
//     }

//     // print_(output_data, 2, 4);

//     for(int i = 0; i<size; i++)
//     {
//         if(kernel_matrix[i*size+i] != 0){
//             preconditioner[i*size+i] = 1/kernel_matrix[i*size+i];
//         }   
//     }
//     // print_(preconditioner, 4, 4);
    
//     preconditioned_conjugate_gradient(kernel_matrix, preconditioner, output_data, approximated_solution, size, dim);

//     // print_(approximated_solution, dim, size);

//     free(kernel_matrix);
//     free(preconditioner);
//     free(output_data);
//     free(approximated_solution);
// }