#ifndef LOCAL_INPUT_SYSTEM_H
#define LOCAL_INPUT_SYSTEM_H


/** \brief This function computes the temporary indices in host that will be used to compute the mapping indices
    \param d_s_array The device array where the temporary indices is stores
    \param map_index The array which stores computed(in this function only boundry points) mapping indices 
    \param stencil_size Size of the stencil
    \param point_count Size of input data
    \param boundry_point_size Size of boundry points
    \param r Number of neighbors for local linear system computation
    \return void
*/
void compute_stencil_array(uint64_t* d_s_array, uint64_t* map_index, int stencil_size, int point_count, int boundry_point_size, int r);

/** \brief This function computes the mapping indices
    \param map_index The array which stores computed mapping indices 
    \param map_index_1 Temprary array that stores the indices of current local input system i.e.(0 0 .. 0 1 1 1 ..1 ... n n n .. n) 
    \param map_index_2 Temprary array that stores the indices of stencil array 
    \param stencil_size Size of the stencil
    \param point_count Size of input data
    \param boundry_point_size Size of boundry points
    \param r Number of neighbors for local linear system computation
    \return void
*/
void compute_map_index(uint64_t* map_index, uint64_t* map_temp_1, uint64_t* map_temp_2, int stencil_size, int point_count, int r);

/** \brief This function computes the local input system i.e. every input data with r nearest points 
    \param localInputData The array that stores local input system containing the r-nearest neighbors
    \param sortedDataPoints Sorted data point
    \param map_index The array that stores the mapping index. 
    \param r Number of neighbors for local linear system computation
    \return void
*/
void compute_local_input_data(double* localInputData, double** sortedDataPoints, uint64_t* map_index, int point_count, int dim, int r);

#endif