#ifndef KERNEL_RIDGE_REGRESSION
#define KERNEL_RIDGE_REGRESSION


// void featch_data(double* input_data, double* output_data, int input_dim, int output_dim, int data_size);
/** \brief Function to build the kernel matrix.
    \param input_data Given input data
    \param kernel_matrix Array that stores the kernel matrix
    \param data_size number of data points
    \param dim dimension of the provided input data
    \return void
*/
void build_matrix(double* input_data, double* kernel_matrix, int data_size, int dim);

/** \brief Function to regularize the kernel matrix
    \param input_data Given input data
    \param hIdMat identity matrix
    \param data_size number of data points
    \return void
*/
void regularize_kernel_matrix(double* kernel_matrix, double* hIdMat, int data_size);

/** \brief Implementation of the kenrel ridge regression algrithm
    \param kernel_matrix Given kernel matrix
    \param preconditioner Given preconditioner matrix
    \param input_data Given input data
    \param output_data Given output data
    \param hIdMat Given identity matrix
    \param testingData Given testing points
    \param predicted_output array that stores the predicted outut of the data
    \param testingDataSize size of testing data points
    \param data_size number of training points
    \param input_dim dimension of input data
    \param output_dim dimesnion of output data
    \return void
*/
void kernel_ridge_regression(double* kernel_matrix, double* preconditioner, double* input_data, double* output_data, double* hIdMat, double* testingData,
                              double* predicted_output, int testingDataSize, int data_size, int input_dim, int output_dim);



#endif