# The implementation of GPU-Parallel Preconditioner for Approximated Krylov-based kernel ridge regression training
**Author: Samundra karki**<br>
**Work: Bachlor's Thesis**<br>
**Title: GPU-Parallel Preconditioner for Approximated Krylov-based kernel ridge regression training**<br><br>
**Instruction** <br>
To compile: **make** <br>
To execute: **./preconditioned_kerenl_ridge_regression**<br><br>
To execute CG using preconditioner using approximate Lagrange basis function:<br>
**Uncomment the "#define LOCAL_PRECONDITIONER" in the main.cu file.**<br><br>
To execute CG using preconditioner tridiagonal precodntioner:<br>
**Uncomment the "#define LU_PRECONDITIONER" in the main.cu file.**<br><br>
To execute CG using unpreconditioner tridiagonal precodntioner:<br>
**Uncomment the "#define UNPRECONDITIONED" in the main.cu file.**<br><br><br>
To change the size of the training inputs:<br>
**Uncomment the number points you want to work with.**<br><br>
To include the testing points: <br>
**Uncomment the section inside // *code* // in generateSyntheticData.cu and kernel_ridge_regression.cu**<br><br><br>
# Thread of exection for preconditioned CG:<br>
**main()** &#8594; **generateData()** &#8594; **mainprecodntioner()** &#8594; **compute_nearest_neighbors()** &#8594; **get_morton_code()**  &#8594; **get_morton_code()** &#8594; **get_morton_ordering()** &#8594; **reorder_point_set()** &#8594; **compute_local_input_data()**  &#8594; **compute_map_index()** &#8594; **compute_local_system_coordinate<<<>>>()** &#8594; **compute_local_kernel_matrix()** &#8594; **kernel_matrix_computation<<<>>>()**  &#8594; **solve_local_linear_system()** &#8594; **construct_preconditioner()** &#8594; **preconditioner_construction<<<>>>()** &#8594; **symmetrise_preconditioner()** &#8594; **build_matrix()** &#8594; **regularize_kernel_matrix()** &#8594; **kernel_ridge_regression()** &#8594; **preconditioned_conjugate_gradient()**<br><br><br>
**The files morton.cu and nearest_neighbors.cu are taken from hmglib library. The author of this code is Prof. Dr. Peter Zaspel.**

