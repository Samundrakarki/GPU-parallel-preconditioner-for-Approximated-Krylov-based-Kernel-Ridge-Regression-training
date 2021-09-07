objects = kernel_ridge_regression.o fetch_data.o gauss_preconditioner.o preconditioned_conjugate_gradient.o nearest_neighbors.o local_input_system.o morton.o local_kernel_matrix.o local_linear_system.o construct_preconditioner.o main_preconditioner.o generateSyntheticData.o main.o

all: $(objects)
	nvcc -lcublas -lcurand $(objects) -o preconditioned_kernel_ridge_regression

%.o: %.cpp
	nvcc -x cu -I. -dc $< -o $@

kernel_ridge_regression.o: kernel_ridge_regression.cu kernel_ridge_regression.h
	nvcc -c kernel_ridge_regression.cu

fetch_data.o: fetch_data.cu fetch_data.h
	nvcc -c fetch_data.cu

gauss_preconditioner.o: gauss_preconditioner.cu gauss_preconditioner.cuh
	nvcc -c gauss_preconditioner.cu

preconditioned_conjugate_gradient.o: preconditioned_conjugate_gradient.cu preconditioned_conjugate_gradient.cuh
	nvcc -c preconditioned_conjugate_gradient.cu

morton.o: morton.cu morton.h
	nvcc -c morton.cu

nearest_neighbors.o: nearest_neighbors.cu nearest_neighbors.cuh
	nvcc -c nearest_neighbors.cu

local_input_system.o: local_input_system.cu local_input_system.cuh
	nvcc -c local_input_system.cu

main_preconditioner.o: main_preconditioner.cu main_preconditioner.h
	nvcc -c main_preconditioner.cu

local_kernel_matrix.o : local_kernel_matrix.cu local_kernel_matrix.cuh
	nvcc -c local_kernel_matrix.cu

local_linear_system.o: local_linear_system.cu local_linear_system.cuh
	nvcc -c local_linear_system.cu

construct_preconditioner.o: construct_preconditioner.cu construct_preconditioner.cuh
	nvcc -c construct_preconditioner.cu

generateSyntheticData.o: generateSyntheticData.cu generateSyntheticData.h
	nvcc -c generateSyntheticData.cu

main.o: main.cu 
	nvcc -c main.cu 

clean:
	rm -rf *.o *.dat preconditioned_kernel_ridge_regression