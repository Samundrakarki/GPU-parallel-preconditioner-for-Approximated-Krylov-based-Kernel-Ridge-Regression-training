#ifndef GAUSS_SEIDEL_PRECONDITIONER
#define GAUSS_SEIDEL_PRECONDITIONER
#include <cuda_runtime.h>

/** \brief Function to compute the preconditoner from the tri-diagoanl matrix.
    \param preconditioner array that stores the preconditioner 
    \param size Number of rows/column of the kernel matrix
    \return void
*/

void gauss_seidel_preconditioner(double* preconditioner, double* kernel_matrix, int size);

#endif