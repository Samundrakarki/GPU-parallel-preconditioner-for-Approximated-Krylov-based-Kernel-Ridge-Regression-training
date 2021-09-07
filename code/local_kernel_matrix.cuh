#ifndef LOCAL_KERNEL_MATRIX_H
#define LOCAL_KERNEL_MATRIX_H

/** \brief This function computes the local kernel matrix.
    \param kernelMatrix Computed kernel matrix will be stored in this array 
    \param localInputData Local input system containing the r-nearest neighbors
    \param dim Dimension of input data
    \param point_count Size of input data
    \param r Number of neighbors for local linear system computation
    \return void
*/
void compute_local_kernel_matrix(double* kernelMatrix, double* localInputData, int dim, int point_count, int r);

#endif