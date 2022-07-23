# The implementation of GPU-Parallel Preconditioner for Approximated Krylov-based kernel ridge regression training
**Author: Samundra karki**<br>
**Work: Bachlor's Thesis**<br>
**Title: GPU-Parallel Preconditioner for Approximated Krylov-based kernel ridge regression training**<br><br>

**Description**<br><br>
The gist of this project is to reduce the time complexity of the krenel ridge regression from O(n^3) to O(n^2) using the conjugate graident method. To further improve the algorithm, we use GPU for parallel computing and the concept of approximate lagrange basis and morton code. The author of the code of the morton code is Peter Zaspel. Futher detials is found in the .pdf document.

**Instruction** <br>
``make`` <br>
``./preconditioned_kerenl_ridge_regression``<br>
CG using preconditioner using approximate Lagrange basis function:
``Uncomment #define LOCAL_PRECONDITIONER``<br>
CG using preconditioner tridiagonal precodntioner:``Uncomment #define LU_PRECONDITIONER``<br>
Exeute CG using unpreconditioner tridiagonal precodntioner:
``Uncomment the "#define UNPRECONDITIONED"``<br>
To change the size of the training inputs:``Uncomment the number points you want to work with``<br>
Include the testing points:``Uncomment the section inside // *code* // in generateSyntheticData.cu and kernel_ridge_regression.cu``

