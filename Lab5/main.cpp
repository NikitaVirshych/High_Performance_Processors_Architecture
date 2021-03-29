#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif

inline
cudaError_t CUDA_CALL(cudaError_t result)
{
	if (result != cudaSuccess)
		std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
	return result;
}

using namespace std;
typedef unsigned char uchar;
typedef unsigned int uint;

class Image {

private:
	uchar* data;
	uint height;
	uint width;
	uint channels;
	uint fullSize;

	void createPad() {
	
		uint height = this->height + 2;
		uint width = this->width + 2 * channels;
		uint fullSize = height * width;

		uchar* tmp = new uchar[fullSize];

		memcpy(&tmp, &data, channels);
		memcpy(&tmp[channels], &data, this->width);
		memcpy(&tmp[channels + this->width], &data[this->width - channels], channels);

		for (int i = 0; i < this->height; i++) {

			memcpy(&tmp[(i + 1) * width], &data[i * this->width], channels);
			memcpy(&tmp[(i + 1) * width + channels], &data[i * this->width], this->width);
			memcpy(&tmp[(i + 1) * width + channels + this->width], &data[(i + 1) * this->width - channels], channels);
		}

		memcpy(&tmp[(this->height + 1) * width], &data[(this->height - 1) * this->width], this->channels);
		memcpy(&tmp[(this->height + 1) * width + channels], &data[(this->height - 1) * this->width], this->width);
		memcpy(&tmp[(this->height + 1) * width + channels + this->width], &data[this->height * this->width - channels], channels);

		delete[] this->data;
		this->data = tmp;
		this->width = width;
		this->height = height;
		this->fullSize = fullSize;
	}

	void removePad() {
	
		uint height = this->height - 2;
		uint width = this->width - 2 * channels;
		uint fullSize = height * width;

		uchar* tmp = new uchar[fullSize];

		for (int i = 0; i < height; i++) {

			memcpy(&tmp[i * width], &data[(i + 1) * this->width + 1], width);
		}

		delete[] this->data;
		this->data = tmp;
		this->width = width;
		this->height = height;
		this->fullSize = fullSize;
	}

public:

	class sizeEx {};

	Image(const char* fileName){

		__loadPPM(
			fileName, &data,
			reinterpret_cast<unsigned int*>(&this->width),
			reinterpret_cast<unsigned int*>(&this->height),
			reinterpret_cast<unsigned int*>(&this->channels)
		);
		this->width *= this->channels;
		this->fullSize = this->width * this->height;
	}

	Image(uint height, uint width, uint channels) : height(height), width(width), channels(channels) {

		this->width *= this->channels;
		this->fullSize = this->width * this->height;

		this->data = new uchar[fullSize];
	}

	Image(const Image& obj) : height(obj.height), width(obj.width), channels(obj.channels), fullSize(obj.fullSize) {

		this->data = new uchar[fullSize];
		memcpy(this->data, obj.data, fullSize);
	}

	~Image() {

		delete[] this->data;
	}

	void save(const char* fileName) {

		__savePPM(fileName, this->data, this->width, this->height, this->channels);
	}

	friend std::ostream& operator<<(std::ostream& outStream, const Image& obj) {

		for (uint i = 0; i < obj.height; i++) {
			for (uint j = 0; j < obj.width; j++)
				outStream << obj.data[j + i * obj.width] << " ";
			outStream << endl;
		}

		return outStream;
	}

	bool operator == (const Image& obj) {

		if (this->fullSize != obj.fullSize)
			return FALSE;

		for (uint i = 0; i < this->height; i++)
			for (uint j = 0; j < this->width; j++)
				if (this->data[i * this->width + j] != obj.data[i * this->width + j])
					return FALSE;

		return TRUE;
	}

	Image filter() {

		Image result(*this);

		this->createPad();

		uchar* elem = this->data + this->width + 1;
		uchar* up = this->data + 1;
		uchar* down = this->data + 2 * this->width + 1;
		int tmp;

		for (uint i = 0; i < result.height; i++) {

			for (uint j = 0; j < result.width; j++) {


				tmp = 5 * (*(elem + j)) - *(up + j) - *(down + j) - *(elem + j - 1) - *(elem + j + 1);

				if (tmp > 255)
					result.data[i * result.width + j] = 255;
				else if (tmp < 0)
					result.data[i * result.width + j] = 0;

			}
		}

		this->removePad();

		return result;
	}

	Image cudaProcessing() const {

	}

};

int main() {

	Image a("kit.pgm");

	Image b = a.filter();

	b.save("Noviy_kit.pgm");

	return 0;
}