#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX 4
#define DIMENSION 10000

#define TILE_SIZE 4

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

__global__ void matrix_mul_kernel(int* a, int* b, int* c)
{
	__shared__ int sharedA[TILE_SIZE][TILE_SIZE];
	__shared__ int sharedB[TILE_SIZE][TILE_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < DIMENSION && col < DIMENSION)
	{
		int result = 0;
		int k;
		int phase;
		
		for (phase = 0; phase <= DIMENSION/TILE_SIZE; phase++)
		{
			sharedA[ty][tx] = a[row * DIMENSION + phase * TILE_SIZE + tx];
			sharedB[ty][tx] = b[(phase * TILE_SIZE + ty) * DIMENSION + col];

			__syncthreads();
			for (k = 0; k < TILE_SIZE; k++)
			{
				if (k + (phase * TILE_SIZE) < DIMENSION) 
				{
					result += (sharedA[ty][k] * sharedB[k][tx]);
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

	int* hostA;
	int* hostB;
	int* hostC;
	int* deviceA;
	int* deviceB;
	int* deviceC;

	hostA = (int*)malloc(size * sizeof(int));
	hostB = (int*)malloc(size * sizeof(int));
	hostC = (int*)malloc(size * sizeof(int));

	int i, j;
	
	initializeMatrix(hostA);
	initializeMatrix(hostB);
    clearMatrix(hostC);

	cudaMalloc((void **)&deviceA, size * sizeof(int));
	cudaMalloc((void **)&deviceB, size * sizeof(int));
	cudaMalloc((void **)&deviceC, size * sizeof(int));

	cudaMemcpy(deviceA, hostA, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, size * sizeof(int), cudaMemcpyHostToDevice);

	int tpg;
	int bpg;

	if(DIMENSION < 100)
	{
		tpg = DIMENSION;
		bpg = 1;
	}
	else
	{
		tpg = TILE_SIZE;
		bpg = (int)ceil(DIMENSION/TILE_SIZE);
	}

	dim3 threadsPerBlock(tpg, tpg);
	dim3 blocksPerGrid(bpg, bpg);

    cudaEventRecord(start);
    matrix_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC);
    cudaEventRecord(stop);

	cudaMemcpy(hostC, deviceC, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	// //print matrix A
    // printf("Matrix A: \n");
	// for (i = 0; i < DIMENSION; i++)
	// {
	// 	for (j = 0; j < DIMENSION; j++)
	// 	{
	// 		printf("%d ", hostA[i * DIMENSION + j]);
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
	// 		printf("%d ", hostB[i * DIMENSION + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");
    //print the resulting matrix
    // printf("Matrix C: \n");
	// for (i = 0; i < DIMENSION; i++)
	// {
	// 	for (j = 0; j < DIMENSION; j++)
	// 	{
	// 		printf("%d ", hostC[i * DIMENSION + j]);
	// 	}
	// 	printf("\n");
	// }
    // printf("\n");
    printf("Time [ms]: %f \n", milliseconds);
}

