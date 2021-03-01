#include <windows.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

class Matrix {

private:
	char* data;
	int height;
	int width;
	int fullSize;

public:

	class sizeEx {};

	Matrix(int height, int width) : height(height), width(width), fullSize(height*width){

		this->data = new char [fullSize];
		ZeroMemory(this->data, fullSize);
		
	}

	Matrix(const Matrix& obj) : height(obj.height), width(obj.width), fullSize(obj.fullSize){

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

		CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

		CURAND_CALL(curandGenerate(gen, devData, this->fullSize/sizeof(unsigned int)));

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
		const int order[] = {2, 1, 3, 0};

		DWORD startTime = GetTickCount();

		for (int h = 0; h < this->height; h++) {
		
			for (int i = 0; i < 4; i++) {
			
				int tmp = order[i];
				for (int j = 0; j < result.width; j++) {
					result.data[(i + h * 4)*result.width + j] = this->data[h * this->width + tmp + j * 4];
				}
			}
		
		}

		cout << "CPU  transform elapsed time: " << GetTickCount() - startTime << " ms";
		
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

		dim3 threadsPerBlock = dim3(result.width / 4);
		dim3 blocksPerGrid = dim3(result.height);

		cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));

		CUDA_CALL(cudaEventRecord(start, 0));

		//data from host to device
		CUDA_CALL(cudaMemcpy(dev_data, this->data, this->fullSize, cudaMemcpyHostToDevice));

		transform <<<blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

		//result from device to host
		CUDA_CALL(cudaMemcpy(result.data, dev_result, this->fullSize, cudaMemcpyDeviceToHost));

		CUDA_CALL(cudaEventRecord(stop, 0));
		CUDA_CALL(cudaEventSynchronize(stop));

		float elapsedTime;
		CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));

		cout << "Cuda transform elapsed time: " << elapsedTime << " ms";

		CUDA_CALL(cudaEventDestroy(start));
		CUDA_CALL(cudaEventDestroy(stop));

		CUDA_CALL(cudaFree(dev_data));
		CUDA_CALL(cudaFree(dev_result));

		return result;
	}

};

__global__ void transform(char* data, char* result) {

	char* window = data + (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	char* rWindow = result + blockIdx.x * blockDim.x * 4 + threadIdx.x;

	for (int i = 0; i < 4; i++) {
	
		*rWindow = *window;
		window++;
		rWindow += blockDim.x * 4;
	}

}

int main() {

	Matrix a(2, 8);

	a.fill();

	cout << a << endl << endl;

	try {
	
		Matrix b = a.cpuTransform();
		cout << b;
	}
	catch (Matrix::sizeEx) {
	
		cout << "Incorrect matrix size";
	}

	return 0;
}