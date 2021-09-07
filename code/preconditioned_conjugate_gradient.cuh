#ifndef PRECONDITIONED_CONJUGATE_GRADIENT
#define PRECONDITIONED_CONJUGATE_GRADIENT

#include <cuda_runtime.h>
#include "cublas_v2.h"

/** \brief This function computes solution to the linear system iteratively
    \param kernel_matrix kernel matrix
    \param preconditioner Preconditioner matrix
    \param output_data Output matrix
    \param approximated_solution Array that stores the approximated solution
    \param data_size Size of input data
    \param output_dim Output dimesnion
    \return void
*/
void preconditioned_conjugate_gradient(double* kernel_matrix, double* preconditioner, double* output_data, 
                                        double* approximated_solution, int data_size, int output_dim);

#endif