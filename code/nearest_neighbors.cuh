#ifndef NEAREST_NEIGHBORS_H
#define NEAREST_NEIGHBORS_H
#include "morton.h"

/** \brief This function computes the nearest neighbors of the given data point
    \param inputData Given input data which is used to build the preconditioner
    \param sortedDataPoints The array that stores sorted data point
    \param point_count Size of input data
    \param dim Dimension of input data
    \param bits Number of bits to be used in Morton computation
    \param r Number of neighbors for local linear system computation
    \return void
*/
void compute_nearest_neighbors(double* inputData, double* outputData, double** sortedDataPoints, double* sorted_inData, 
    double* sorted_outData, uint64_t* order, int point_count, int dim, int bits, int r);
// void compute_nearest_neighbors(double* inputData, double** sortedDataPoints, double* sorted_inData, uint64_t* order, int point_count, int dim, int bits, int r);

#endif
