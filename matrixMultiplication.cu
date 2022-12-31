/*
CUDA - generate array of random numbers and calculate occurence of odd and even numbers - no streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 4
#define DIMENSION 1000

 void initializeMatrix( int* matrix) {
    srand(time(0));
	for (int i = 0; i < DIMENSION; i++)
	{
		for (int j = 0; j < DIMENSION; j++) 
		{
			matrix[i * DIMENSION + j] = rand() % MAX;
		}
	}
}
void clearMatrix( int* matrix) {
	for (int i = 0; i < DIMENSION; i++)
	{
		for (int j = 0; j < DIMENSION; j++) 
		{
			matrix[i * DIMENSION + j] = 0;
		}
	}
}

//cuda kernel for multiplying two matrices without tiling
__global__ void matrix_mul_kernel(int* a, int* b, int* c)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//check if thread directly maps to the dimensions of resulting matrix
	if (row < DIMENSION && col < DIMENSION)
	{
		int result = 0;
		int k;
		for (k = 0; k < DIMENSION; k++)
		{
			result += (a[row * DIMENSION + k] * b[k * DIMENSION + col]);
		}
		c[row * DIMENSION + col] = result;
	}
}


int main(int argc, char **argv)
{
    int size = DIMENSION * DIMENSION;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	//declare host and device matrices pointers
	int* mat_a;
	int* mat_b;
	int* mat_c;
	int* d_mat_a;
	int* d_mat_b;
	int* d_mat_c;
	
	//allocate memory for host matrices
	mat_a = (int*)malloc(size * sizeof(int));
	mat_b = (int*)malloc(size * sizeof(int));
	mat_c = (int*)malloc(size * sizeof(int));
	
	int i, j;
	
	initializeMatrix(mat_a);
	initializeMatrix(mat_b);
    clearMatrix(mat_c);

	//allocate matrices memeory on device
	cudaMalloc((void **)&d_mat_a, size * sizeof(int));
	cudaMalloc((void **)&d_mat_b, size * sizeof(int));
	cudaMalloc((void **)&d_mat_c, size * sizeof(int));

	//copy A and B matrices from host to device
	cudaMemcpy(d_mat_a, mat_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, size * sizeof(int), cudaMemcpyHostToDevice);

	//execute cuda kernel
    dim3 threadsPerBlock(DIMENSION, DIMENSION);
    dim3 blocksPerGrid(1, 1);
        if (size > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(DIMENSION)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(DIMENSION)/double(threadsPerBlock.y));
        }

    cudaEventRecord(start);
    matrix_mul_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_mat_a, d_mat_b, d_mat_c);
    cudaEventRecord(stop);

	//copy the compute matrix C from device to host
	cudaMemcpy(mat_c, d_mat_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	
	//free cuda memory
	cudaFree(d_mat_a);
	cudaFree(d_mat_b);
	cudaFree(d_mat_c);

	//print matrix A
    printf("Matrix A: \n");
	for (i = 0; i < DIMENSION; i++)
	{
		for (j = 0; j < DIMENSION; j++)
		{
			printf("%d ", mat_a[i * DIMENSION + j]);
		}
		printf("\n");
	}
    printf("\n");
    //print matrix B
    printf("Matrix B: \n");
	for (i = 0; i < DIMENSION; i++)
	{
		for (j = 0; j < DIMENSION; j++)
		{
			printf("%d ", mat_b[i * DIMENSION + j]);
		}
		printf("\n");
	}
    printf("\n");
    //print the resulting matrix
    printf("Matrix C: \n");
	for (i = 0; i < DIMENSION; i++)
	{
		for (j = 0; j < DIMENSION; j++)
		{
			printf("%d ", mat_c[i * DIMENSION + j]);
		}
		printf("\n");
	}
    printf("\n");
    printf("Time [ms]: %f \n", milliseconds);
}

