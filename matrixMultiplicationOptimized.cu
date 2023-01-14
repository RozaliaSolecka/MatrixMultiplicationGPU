/*
CUDA - generate array of random numbers and calculate occurence of odd and even numbers - no streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 4
#define DIMENSION 10000
//size of the share memory tile in the device
#define TILE_SIZE 16

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
	//declare shared memory matrices for A and B matrices
	__shared__ int shared_a_tile[TILE_SIZE][TILE_SIZE];
	__shared__ int shared_b_tile[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	//check if thread directly maps to the dimensions of resulting matrix
	if (row < DIMENSION && col < DIMENSION)
	{
		int result = 0;
		int k;
		int phase;
		
		//calculate C matrix indexes in phases. Each phase shares 
		//TILE_SIZE * TILE_SIZE data copied to the shared matrix A 
		//and matrix B.
		for (phase = 0; phase <= DIMENSION/TILE_SIZE; phase++)
		{
			shared_a_tile[ty][tx] = a[row * DIMENSION + phase * TILE_SIZE + tx];
			shared_b_tile[ty][tx] = b[(phase * TILE_SIZE + ty) * DIMENSION + col];
			__syncthreads();
			
			for (k = 0; k < TILE_SIZE; k++)
			{
				if (k + (phase * TILE_SIZE) < DIMENSION) 
				{
					result += (shared_a_tile[ty][k] * shared_b_tile[k][tx]);
				}
			}
			__syncthreads();
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

	//declare dimensions for the grid and block
	dim3 dimBlock(TILE_SIZE,TILE_SIZE);
	dim3 dimGrid((int)ceil(DIMENSION/TILE_SIZE),(int)ceil(DIMENSION/TILE_SIZE));


    cudaEventRecord(start);
    matrix_mul_kernel<<<dimGrid, dimBlock>>>(d_mat_a, d_mat_b, d_mat_c);
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

	// //print matrix A
    // printf("Matrix A: \n");
	// for (i = 0; i < DIMENSION; i++)
	// {
	// 	for (j = 0; j < DIMENSION; j++)
	// 	{
	// 		printf("%d ", mat_a[i * DIMENSION + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");
    // //print matrix B
    // printf("Matrix B: \n");
	// for (i = 0; i < DIMENSION; i++)
	// {
	// 	for (j = 0; j < DIMENSION; j++)
	// 	{
	// 		printf("%d ", mat_b[i * DIMENSION + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");
    // //print the resulting matrix
    // printf("Matrix C: \n");
	// for (i = 0; i < DIMENSION; i++)
	// {
	// 	for (j = 0; j < DIMENSION; j++)
	// 	{
	// 		printf("%d ", mat_c[i * DIMENSION + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");
    printf("Time [ms]: %f \n", milliseconds);
}

