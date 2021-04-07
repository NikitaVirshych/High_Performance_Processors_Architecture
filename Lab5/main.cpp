#include "Header.h"
#include "kernel.cu"

using namespace std;
typedef unsigned char uchar;
typedef unsigned int uint;

const unsigned int PGMHeaderSize = 0x40;

class Image {

private:
	uchar* data = nullptr;
	uint height = 0;
	uint width = 0;
	uint channels = 0;
	uint fullSize = 0;

public:

	void createPad()
	{
		uint height = this->height + 2;
		uint width = this->width + 2 * channels;
		uint fullSize = height * width;

		uchar* tmp = new uchar[fullSize];
		ZeroMemory(tmp, fullSize);

		memcpy(tmp, data, channels);
		memcpy(&tmp[channels], data, this->width);
		memcpy(&tmp[channels + this->width], &data[this->width - channels], channels);

		for (int i = 0; i < this->height; i++)
		{
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

	void removePad()
	{
		uint height = this->height - 2;
		uint width = this->width - 2 * channels;
		uint fullSize = height * width;

		uchar* tmp = new uchar[fullSize];

		for (int i = 0; i < height; i++)
		{

			memcpy(&tmp[i * width], &data[(i + 1) * this->width + channels], width);
		}

		delete[] this->data;
		this->data = tmp;
		this->width = width;
		this->height = height;
		this->fullSize = fullSize;
	}

	class sizeEx {};

	Image() {}

	Image(uint height, uint width, uint channels) : height(height), width(width), channels(channels) 
	{
		this->width *= this->channels;
		this->fullSize = this->width * this->height;

		this->data = new uchar[fullSize];
	}

	Image(const Image& obj) : height(obj.height), width(obj.width), channels(obj.channels), fullSize(obj.fullSize) 
	{
		this->data = new uchar[fullSize];
		memcpy(this->data, obj.data, fullSize);
	}

	~Image() 
	{
		delete[] this->data;
	}

	bool load(const char* fileName)
	{
		FILE* fp = fopen(fileName, "rb");
		if (!fp)
		{
			cerr << "Failed to open input file: " << fileName << endl;
			return false;
		}

		// check header
		char header[PGMHeaderSize];
		if (fgets(header, PGMHeaderSize, fp) == NULL)
		{
			cerr << "Reading PGM header returned NULL" << endl;
			return false;
		}
		if (strncmp(header, "P5", 2) == 0)
		{
			this->channels = 1;
		}
		else if (strncmp(header, "P6", 2) == 0)
		{
			this->channels = 3;
		}
		else
		{
			cerr << "Input file is not a PPM or PGM image" << endl;
			return false;
		}

		// parse header, read maxval, width and height
		unsigned int maxval = 0;
		unsigned int i = 0;
		while (i < 3)
		{
			if (fgets(header, PGMHeaderSize, fp) == NULL) {
				std::cerr << "Reading PGM header returned NULL" << std::endl;
				return false;
			}
			if (header[0] == '#')
				continue;

			if (i == 0)
			{
				i += sscanf(header, "%u %u %u", &this->width, &this->height, &maxval);
			}
			else if (i == 1)
			{
				i += sscanf(header, "%u %u", &this->height, &maxval);
			}
			else if (i == 2)
			{
				i += sscanf(header, "%u", &maxval);
			}
		}

		this->width *= this->channels;
		this->fullSize = this->width * this->height;

		data = new uchar[fullSize];

		// read and close file
		if (fread(data, sizeof(unsigned char), fullSize, fp) == 0)
		{
			cerr << "Read data returned error." << endl;
			return false;
		}
		fclose(fp);

		return true;
	}

	bool save(const char* fileName) const
	{
		fstream fh(fileName, fstream::out | fstream::binary);

		if (fh.bad())
		{
			cerr << "Opening output file failed." << endl;
			return false;
		}

		if (channels == 1)
		{
			fh << "P5\n";
		}
		else if (channels == 3) {
			fh << "P6\n";
		}
		else {
			cerr << "Invalid number of channels." << endl;
			return false;
		}

		fh << this->width / channels << "\n" << this->height << "\n" << 0xff << endl;

		for (unsigned int i = 0; i < this->fullSize && fh.good(); ++i)
		{
			fh << data[i];
		}
		fh.flush();

		if (fh.bad())
		{
			cerr << "Writing data failed." << endl;
			return false;
		}
		fh.close();

		return true;
	}

	friend std::ostream& operator<<(std::ostream& outStream, const Image& obj) 
	{
		for (uint i = 0; i < obj.height; i++) 
		{
			for (uint j = 0; j < obj.width; j++)
				outStream << setw(3) << (int)obj.data[j + i * obj.width] << " ";
			outStream << endl;
		}

		return outStream;
	}

	bool operator == (const Image& obj) 
	{
		if (this->fullSize != obj.fullSize)
			return FALSE;

		for (uint i = 0; i < this->height; i++)
			for (uint j = 0; j < this->width; j++)
				if (this->data[i * this->width + j] != obj.data[i * this->width + j])
					return FALSE;

		return TRUE;
	}

	Image convolute() 
	{
		Image result(*this);

		this->createPad();

		uchar* elem = this->data + this->width + channels;
		uchar* up = this->data + channels;
		uchar* down = this->data + 2 * this->width + channels;
		int tmp;

		for (uint i = 0; i < result.height; i++) 
		{
			for (uint j = 0; j < result.width; j++) 
			{
				tmp = 5 * (*(elem + j)) - *(up + j) - *(down + j) - *(elem + j - channels) - *(elem + j + channels);

				if (tmp > 255)
					result.data[i * result.width + j] = 255;
				else if (tmp < 0)
					result.data[i * result.width + j] = 0;
				else
					result.data[i * result.width + j] = tmp;
			}
			elem += this->width;
			up += this->width;
			down += this->width;
		}

		this->removePad();

		return result;
	}

	Image cudaConvolute() 
	{

		Image result(*this);

		this->createPad();

		dim3 threadsPerBlock = dim3(32, 32);
		dim3 blocksPerGrid = dim3(this.height / 31);
		int lastBlockRows = 32;

		if (this.height % 31) 
		{		
			lastBlockRows = this.height % 30;
			blocksPerGrid = dim3((this.height / 31) + 1)
		}

		int pitch = this->width / 128;
		if (this->width % 128)
			pitch += 1;
		pitch *= 128;

		int* dev_data;
		int* dev_result;

		cudaMalloc(&dev_data, pitch * this->height);
		cudaMalloc(&dev_result, pitch * result.height);

		cudaMemcpy2D(dev_data, pitch, this->data, this->width, this->width, this->height, cudaMemcpyHostToDevice);

		int channels = this->channels;
		cudaMemcpyToSymbol(CHANNELS, &channels, sizeof(int));
		cudaMemcpyToSymbol(LAST_ROWS, &lastBlockRows, sizeof(int));
		cudaMemcpyToSymbol(PITCH, &pitch, sizeof(int));
		int transactions = pitch / 128;
		cudaMemcpyToSymbol(TRANSACTIONS, &transactions, sizeof(int));

		convolution <<< blocksPerGrid, threadsPerBlock >>> (dev_data, dev_result);

		cudaMemcpy2D(result.data, result.width, dev_result, pitch, result.width, result.height, cudaMemcpyDeviceToHost);

		this->removePad();

		return result;

	}

};

int main() {

	Image a;

	if (a.load("kit.pgm"))
		cout << "load successful" << endl;

	Image b = a.convolute();

	if (b.save("Noviy_kit.pgm"))
		cout << "save successful";

	return 0;
}