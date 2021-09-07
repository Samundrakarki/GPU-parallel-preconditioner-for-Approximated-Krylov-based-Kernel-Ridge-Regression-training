
#ifndef MAIN_PRECONDITIONER
#define MAIN_PRECONDITIONER

/** \brief Function to call the required function to construct a precontioner.
    \param inputData Given input data to be sorted. This is used to build the preconditioner
    \param outputData Given output data to be sorted
    \param sorted_inData An array that will store the sorted input data 
    \param sorted_outData An array that will store the sorted output data 
    \param precondtioner A preconditioner array which would be computed by this function
    \param dataSize Size of input data
    \param dim Dimension of input data
    \param r Number of neighbors for local linear system computation
    \param bits Number of bits to be used in Morton computation
    \return void
*/
void main_preconditioner(double* inputData, double* outputData, double* sorted_inData, double* sorted_outData, 
                            double* preconditioner, int dataSize, int dim, int r, int bits);
// void main_preconditioner(double* inputData, double* sorted_inData, uint64_t* order, double* preconditioner, int dataSize, int dim, int r, int bits);

/** \brief Function to print passed matrix.
    \param d_matrix Matrix to be printed 
    \param nRows Number of rows of the matrix
    \param nColumns Number of columns of the matrix
    \return void
*/
void print_matrix(double* d_matrix, int nRows, int nColumns);

#endif