#ifndef LOCAL_LINEAR_SYSTEM_H
#define LOCAL_LINEAR_SYSTEM_H

/** \brief This function solves every local linear system.
    \param localKernelMatrix Kernel matrix that consists all of the local linear systems
    \param localSolution The array that will store the solution to local linear systems.
    \param dataSize Size of input data
    \param r Number of neighbors for local linear system computation
    \return void
*/
void solve_local_linear_system(double* localKernelMatrix, double* localSolution, int dataSize, int r);

#endif