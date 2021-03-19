#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>

__device__ __constant__ int ORDER[4];
__device__ __constant__ int WIDTH;

__global__ void transform(char* data, char* result) {

	result[blockIdx.y * WIDTH + ORDER[threadIdx.x % 4] * WIDTH / 4 + blockDim.x * blockIdx.x / 4 + threadIdx.x / 4] = data[blockIdx.y * WIDTH + blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void sharedTransform(int* data, int* result) {

	__shared__ char memory[4][128];
	char buffer[4];

	*(int*)buffer = data[blockIdx.y * WIDTH + blockIdx.x * WIDTH / gridDim.x + threadIdx.y * blockDim.x + threadIdx.x];

	memory[3][32 * threadIdx.y + threadIdx.x] = buffer[0];
	memory[1][32 * threadIdx.y + threadIdx.x] = buffer[1];
	memory[0][32 * threadIdx.y + threadIdx.x] = buffer[2];
	memory[2][32 * threadIdx.y + threadIdx.x] = buffer[3];

	__syncthreads();

	result[blockIdx.y * WIDTH + blockIdx.x * WIDTH / 4 / gridDim.x + threadIdx.y * WIDTH / 4 + threadIdx.x] = *(int*)(memory[threadIdx.y] + 4 * threadIdx.x);
}

inline
cudaError_t CUDA_CALL(cudaError_t result)
{
	if (result != cudaSuccess)
		std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
	return result;
}
inline
curandStatus_t CURAND_CALL(curandStatus_t result)
{
	if (result != CURAND_STATUS_SUCCESS)
		std::cerr << "CURAND Runtime Error: " << std::endl;
	return result;
}

using namespace std;

class Matrix {

private:
	char* data;
	int height;
	int width;
	int fullSize;

public:

	class sizeEx {};

	Matrix(int height, int width) : height(height), width(width), fullSize(height* width) {

		this->data = new char[fullSize];
		ZeroMemory(this->data, fullSize);

	}

	Matrix(const Matrix& obj) : height(obj.height), width(obj.width), fullSize(obj.fullSize) {

		this->data = new char[fullSize];
		memcpy(this->data, obj.data, fullSize);
	}

	~Matrix() {

		delete[] this->data;
	}

	void fill() {
		for (int i = 0; i < fullSize; i++)
			this->data[i] = '0' + rand() % 10;
	}

	void cudaFill() {

		curandGenerator_t gen;
		char* devData;

		CUDA_CALL(cudaMalloc(&devData, this->fullSize));

		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

		CURAND_CALL(curandGenerate(gen, (unsigned int*)devData, this->fullSize / sizeof(unsigned int)));

		CUDA_CALL(cudaMemcpy(this->data, devData, this->fullSize, cudaMemcpyDeviceToHost));
		CURAND_CALL(curandDestroyGenerator(gen));

		CUDA_CALL(cudaFree(devData));
	}

	friend std::ostream& operator<<(std::ostream& outStream, const Matrix& obj) {

		for (int i = 0; i < obj.height; i++) {
			for (int j = 0; j < obj.width; j++)
				outStream << obj.data[j + i * obj.width] << " ";
			outStream << endl;
		}

		return outStream;
	}

	bool operator == (const Matrix& obj) {

		if (this->fullSize != obj.fullSize)
			return FALSE;

		for (int i = 0; i < this->height; i++)
			if (memcmp(this->data, obj.data, fullSize))
				return FALSE;

		return TRUE;
	}

	Matrix cpuTransform() const {

		if (this->width % 4)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);
		const int order[] = { 2, 1, 3, 0 };

		DWORD64 startTime = GetTickCount64();

		for (int h = 0; h < this->height; h++) {

			for (int i = 0; i < 4; i++) {

				int tmp = order[i];
				for (int j = 0; j < result.width; j++) {
					result.data[(i + h * 4) * result.width + j] = this->data[h * this->width + tmp + j * 4];
				}
			}

		}

		cout << "CPU  transform elapsed time: " << GetTickCount64() - startTime << " ms" << endl;

		return result;
	}

	Matrix cudaTransform() const {

		if (this->width % 128)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);

		char* dev_data;
		char* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		dim3 threadsPerBlock = dim3(128);
		dim3 blocksPerGrid = dim3(this->width / 128, this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(WIDTH, &this->width, sizeof(int)));
		int order[] = { 3, 1, 0, 2 };
		CUDA_CALL(cudaMemcpyToSymbol(ORDER, order, sizeof(int) * 4));


		transform << < blocksPerGrid, threadsPerBlock >> > (dev_data, dev_result);

		//result from device to host
		CUDA_CALL(cudaMemcpy(result.data, dev_result, this->fullSize, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));

		float elapsedTime;
		CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

		cout << "Cuda transform elapsed time: " << (int)elapsedTime << " ms" << endl;

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		CUDA_CALL(cudaFree(dev_data));
		CUDA_CALL(cudaFree(dev_result));

		return result;
	}

	Matrix cudaSharedTransform() const {

		if (this->width % 512)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);

		int* dev_data;
		int* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		int intWidth = this->width / sizeof(int);
		int warps = 4;

		dim3 threadsPerBlock = dim3(32, warps);
		dim3 blocksPerGrid = dim3(intWidth / (32 * warps), this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(WIDTH, &intWidth, sizeof(int)));

		sharedTransform <<< blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

		//result from device to host
		CUDA_CALL(cudaMemcpy(result.data, dev_result, this->fullSize, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));

		float elapsedTime;
		CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

		cout << "Cuda STransform elapsed time: " << (int)elapsedTime << " ms" << endl;

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		CUDA_CALL(cudaFree(dev_data));
		CUDA_CALL(cudaFree(dev_result));

		return result;
	}

	Matrix cudaPinnedTransform() const {

		if (this->width % 512)
			throw sizeEx();

		CUDA_CALL(cudaHostRegister(this->data, this->fullSize, cudaHostRegisterDefault));

		Matrix result(this->height * 4, this->width / 4);

		int* dev_data;
		int* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		int intWidth = this->width / sizeof(int);
		int warps = 4;

		dim3 threadsPerBlock = dim3(32, warps);
		dim3 blocksPerGrid = dim3(intWidth / (32 * warps), this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(WIDTH, &intWidth, sizeof(int)));

		sharedTransform <<< blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

		//result from device to host
		CUDA_CALL(cudaMemcpy(result.data, dev_result, this->fullSize, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));

		float elapsedTime;
		CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

		cout << "Cuda STransform elapsed time: " << (int)elapsedTime << " ms" << endl;

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		CUDA_CALL(cudaFree(dev_data));
		CUDA_CALL(cudaFree(dev_result));

		return result;
	}

	Matrix cudaBigTransform() const {

		if (this->width % 512)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);
		int* dev_data;
		int* dev_result;
		int* pinned;

		int intWidth = this->width / sizeof(int);
		int warps = 4;
		dim3 threadsPerBlock = dim3(32, warps);
		dim3 blocksPerGrid;

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));
		float fullTime, elapsedTime;

		size_t freeMem;
		int rows, offset = 0, rowsComplete = 0;

		while(rowsComplete < this->height){

			CUDA_CALL(cudaMemGetInfo(&freeMem, nullptr));
			rows = freeMem / this->width;
		
			if(rows > this->height - rowsComplete)
				rows = this->height - rowsComplete;

			cout << "Processing " << rows << " rows" << endl;

			dim3 blocksPerGrid = dim3(intWidth / (32 * warps), rows);

			CUDA_CALL(cudaMalloc(&dev_data, rows * this->width));
			CUDA_CALL(cudaMalloc(&dev_result, rows * this->width));

			CUDA_CALL(cudaMallocHost(&pinned, rows * this->width));

			CUDA_CALL(cudaEventRecord(start));

			CUDA_CALL(cudaMemcpy(pinned, this->data + offset, rows * this->width, cudaMemcpyHostToHost));

			//data from host to device
			CUDA_CALL(cudaMemcpy(dev_data, pinned, rows * this->width, cudaMemcpyHostToDevice));
			
			CUDA_CALL(cudaMemcpyToSymbol(WIDTH, &intWidth, sizeof(int)));
	
			sharedTransform <<< blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

			//result from device to host
			CUDA_CALL(cudaMemcpy(pinned, dev_result, rows * this->width, cudaMemcpyDeviceToHost));

			CUDA_CALL(cudaMemcpy(result.data + offset, pinned, rows * this->width, cudaMemcpyHostToHost));
	
			CUDA_CALL(cudaEventRecord(stop));
			CUDA_CALL(cudaEventSynchronize(stop));

			CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
			fullTime += elapsedTime;

			rowsComplete += rows;
			offset += rows * this->width;

			cout << rowsComplete << " rows out of " << this->height << " processed" << endl;

			CUDA_CALL(cudaFree(dev_data));
			CUDA_CALL(cudaFree(dev_result));
			CUDA_CALL(cudaFree(pinned));
		}	

		cout << "Cuda BTransform elapsed time: " << (int)elapsedTime << " ms" << endl;	

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		return result;
	}

	void printSubmatrix(int x0, int y0, int x1, int y1) const {

		if (x0 > x1 || y0 > y1)
			return;

		if (x1 - x0 > this->width || y1 - y0 > this->height)
			throw sizeEx();

		for (int i = y0 - 1; i < y1; i++) {
			for (int j = x0 - 1; j < x1; j++)
				cout << this->data[j + i * this->width] << " ";
			cout << endl;
		}
	}

};

#define HEIGHT 5000
#define WIDTH_AMP 100

int main() {

	Matrix a(HEIGHT, WIDTH_AMP * 512);

	a.fill();

	try {
		Matrix b = a.cpuTransform();
		Matrix c = a.cudaTransform();
		Matrix d = a.cudaSharedTransform();
		Matrix e = a.cudaPinnedTransform();


		if (d == b && b == c)
			cout << "vse ok";
		else 
			cout << "ne vse ok";
	}
	catch (Matrix::sizeEx) {

		cout << "Incorrect matrix size";
	}

	return 0;
}