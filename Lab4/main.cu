
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>

__global__ void transform(char* data, char* result) {

	char* window = data + (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	char* rWindow = result + blockIdx.x * blockDim.x * 4 + threadIdx.x;

	rWindow[0] = window[2];
	rWindow[blockDim.x] = window[1];
	rWindow[blockDim.x * 2] = window[3];
	rWindow[blockDim.x * 3] = window[0];


}

__device__ __constant__ int ORDER[4];
__device__ __constant__ int WIDTH;

__global__ void transform2(char* data, char* result) {

	result[blockIdx.y * WIDTH + ORDER[threadIdx.x % 4] * WIDTH / 4 + blockDim.x * blockIdx.x / 4 + threadIdx.x / 4] = data[blockIdx.y * WIDTH + blockIdx.x * blockDim.x + threadIdx.x];
}

__global__ void sharedTransform(char* data, char* result) {

	__shared__ char* memory;

	if (!threadIdx.x) {		
		memory = (char*)malloc(WIDTH);
	}

	__syncthreads();

	for (int i = 0; i < gridDim.x; i++) {

		memory[ORDER[threadIdx.x % 4] * WIDTH / 4 + blockDim.x * i / 4 + threadIdx.x / 4] = data[blockIdx.x * WIDTH + i * blockDim.x + threadIdx.x];
	}

	__syncthreads();


	for (int i = 0; i < gridDim.x; i++) {

		result[blockIdx.x * WIDTH + i * blockDim.x  + threadIdx.x] = memory[i * blockDim.x + threadIdx.x];
	}
	
	__syncthreads();
	
	if (!threadIdx.x) {
	
		free(memory);
	}
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
		std::cerr << "CUDA Runtime Error: " << std::endl;
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

		if (this->width % 4)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);

		char* dev_data;
		char* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		dim3 threadsPerBlock = dim3(result.width);
		dim3 blocksPerGrid = dim3(this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));

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

	Matrix cudaTransform2() const {

		if (this->width % 128)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);

		char* dev_data;
		char* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		dim3 threadsPerBlock = dim3(128);
		dim3 blocksPerGrid = dim3(this->width/128, this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(WIDTH, this->width, sizeof(int));
		int order[] = { 3, 1, 0, 2 };
		CUDA_CALL(cudaMemcpyToSymbol(ORDER, order, sizeof(int)*4);
		

		transform2 <<< blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

		//result from device to host
		CUDA_CALL(cudaMemcpy(result.data, dev_result, this->fullSize, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));

		float elapsedTime;
		CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

		cout << "Cuda transform2 elapsed time: " << (int)elapsedTime << " ms" << endl;

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		CUDA_CALL(cudaFree(dev_data));
		CUDA_CALL(cudaFree(dev_result));

		return result;
	}

	Matrix cudaSharedTransform() const {

		if (this->width % 128)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);

		char* dev_data;
		char* dev_result;

		CUDA_CALL(cudaMalloc(&dev_data, this->fullSize));
		CUDA_CALL(cudaMalloc(&dev_result, result.fullSize));

		dim3 threadsPerBlock = dim3(128);
		dim3 blocksPerGrid = dim3(this->height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(WIDTH, this->width, sizeof(int));
		int order[] = { 3, 1, 0, 2 };
		CUDA_CALL(cudaMemcpyToSymbol(ORDER, order, sizeof(int)*4);
		
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
	
	void printSubmatrix(int x0, int y0, int x1, int y1) const {

		if(x0 > x1 || y0 > y1)
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

#define HEIGHT 50000
#define WIDTH_AMP 8

int main() {

	Matrix a(HEIGHT, WIDTH_AMP * 128);

	a.cudaFill();

	try {
		Matrix b = a.cudaTransform();
		Matrix c = a.cudaTransform2();
		Matrix d = a.cudaSharedTransform();

		if (b == c || b == d)
			cout << "vse ok";
		else cout << "ne vse ok";
	}
	catch (Matrix::sizeEx) {

		cout << "Incorrect matrix size";
	}

	return 0;
}