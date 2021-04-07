#include "Header.h"

__device__ __constant__ int CHANNELS;
__device__ __constant__ int LAST_ROWS;
__device__ __constant__ int PITCH;
__device__ __constant__ int TRANSACTIONS;

__global__ void convolution(int* data, int* result) 
{
	int rows = blockDim.y;

	if(blockIdx.x == gridDim.x)
		rows = LAST_ROWS;

	__shared__ unsigned char localData[rows][256];
	unsigned char localResult[4];
	//int tmp;

	if(threadIdx.y < rows)
	{
		*(int*)(localData[threadIdx.y] + 4 * threadIdx.x) = data[PITCH*31*blockIdx.x + PITCH * threadIdx.y + threadIdx.x];

		for(int i = 1; i < TRANSACTIONS; i++)
		{
			//Новая транзакция
			*(int*)(localData[threadIdx.y] + 128 + 4 * threadIdx.x) = data[PITCH*31*blockIdx.x + PITCH * threadIdx.y + 32 * i + threadIdx.x];
	
			__syncthreads();

			if(threadIdx.y < rows - 2)
			{
				for(int j = 0; j < 4; j++)
					localResult[j] = localData[threadIdx.y + 1][CHANNELS + 4 * threadIdx.x + j];
				
				__syncthreads();

				result[PITCH*30*blockIdx.x + PITCH * threadIdx.y + 32 * (i-1) + threadIdx.x] = *(int*)localResult;
			}

			__syncthreads();

			//Сдвиг в разделяемой памяти
			*(int*)localResult = *(int*)(localData[threadIdx.y] + 128 + 4 * threadIdx.x);
			*(int*)(localData[threadIdx.y] + 4 * threadIdx.x) = *(int*)localResult;

		}
	
	}	
}


/*

tmp = 5 * (*(elem + j)) - *(up + j) - *(down + j) - *(elem + j - channels) - *(elem + j + channels);

				if (tmp > 255)
					result.data[i * result.width + j] = 255;
				else if (tmp < 0)
					result.data[i * result.width + j] = 0;
				else
					result.data[i * result.width + j] = tmp;

*/		