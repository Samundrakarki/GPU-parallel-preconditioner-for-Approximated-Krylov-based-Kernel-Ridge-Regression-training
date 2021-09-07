#include <stdint.h>
#ifndef CONSTRUCT_PRECONDITIONER_H
#define CONSTRUCT_PRECONDITIONER_H

/** \brief This function constructs the preconditioner using local solution from input data.
    \param dpreconditioner The device array that stores the preconditionr
    \param dLocalSolution The device array od the solution of the local system
    \param dMappingIndex The device mapping indices 
    \param dataSize Size of input data
    \param r Number of neighbors for local linear system computation
    \return void
*/
void construct_preconditioner(double* dpreconditioner, double* dLocalSolution, uint64_t* dMappingIndex, uint64_t* order, int dataSize, int r);

#endif