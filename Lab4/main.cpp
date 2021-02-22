#include <windows.h>
#include <iostream>
#include <iomanip>
#include <intrin.h>

#pragma intrinsic(__rdtsc)

using namespace std;

class Matrix {

private:
	double** matrix;
	int height;
	int width;

public:

	class sizeEx {};

	Matrix(int height, int width) : height(height), width(width) {

		this->matrix = new double* [this->height];
		for (int i = 0; i < this->height; i++) {
			this->matrix[i] = new double[this->width];
			ZeroMemory(this->matrix[i], this->width * sizeof(double));
		}
	}

	Matrix(const Matrix& obj) : height(obj.height), width(obj.width) {

		this->matrix = new double* [this->height];
		for (int i = 0; i < this->height; i++) {
			this->matrix[i] = new double[this->width];
			memcpy(this->matrix[i], obj.matrix[i], this->width * sizeof(double));
		}
	}

	~Matrix() {

		for (int i = 0; i < this->height; i++)
			delete[] this->matrix[i];
		delete[] this->matrix;
	}

	void fill() {
		for (int i = 0; i < this->height; i++)
			for (int j = 0; j < this->width; j++)
				this->matrix[i][j] = rand() % 5;
	}

	double* operator [](int index) { return this->matrix[index]; }

	friend std::ostream& operator<<(std::ostream& outStream, const Matrix& obj) {

		for (int i = 0; i < obj.height; i++) {
			for (int j = 0; j < obj.width; j++)
				outStream << setw(3) << obj.matrix[i][j] << " ";
			outStream << endl;
		}

		return outStream;
	}

	bool operator == (const Matrix& obj) {

		if (this->height != obj.height || this->width != obj.width)
			return FALSE;

		for (int i = 0; i < this->height; i++)
			if (memcmp(this->matrix[i], obj.matrix[i], this->width * sizeof(double)))
				return FALSE;

		return TRUE;
	}

	Matrix transform() const {
	
		if (this->width % 4)
			throw sizeEx();

		Matrix result(this->height * 4, this->width / 4);
		const int order[] = {2, 1, 3, 0};

		for (int h = 0; h < this->height; h++) {
		
			for (int i = 0; i < 4; i++) {
			
				int tmp = order[i];
				for (int j = 0; j < result.width; j++) {
					result.matrix[i + h * 4][j] = this->matrix[h][tmp + j * 4];
				}
			}
		
		}
		
		return result;
	}

};


int main() {

	Matrix a(2, 8);

	a.fill();

	cout << a << endl << endl;

	try {
	
		Matrix b = a.transform();
		cout << b;
	}
	catch (Matrix::sizeEx) {
	
		cout << "Incorrect matrix size";
	}

	return 0;
}