#include <iostream>
#include <iomanip>
#include <emmintrin.h>
#include <windows.h>

using namespace std;

class Matrix {

private:
	double** matrix;
	int height;
	int width;

public:

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

	Matrix operator * (const Matrix& obj) {

		int resultWidth = obj.width;

		Matrix result(this->height, resultWidth);
		DWORD startTime = GetTickCount();

		for (int i = 0; i < this->height; i++) {

			double* resultRow = result[i];

			for (int k = 0; k < this->width; k++) {

				const double* objRow = obj.matrix[k];
				double thisElem = this->matrix[i][k];

				for (int j = 0; j < resultWidth; j++)
					resultRow[j] += thisElem * objRow[j];
			}
		}

		cout << setw(9) << "* : " << GetTickCount() - startTime << " milliseconds" << endl;

		return result;
	}

	Matrix noVecMul (const Matrix& obj) {

		int resultWidth = obj.width;

		Matrix result(this->height, obj.width);
		DWORD startTime = GetTickCount();

		for (int i = 0; i < this->height; i++) {

			double* resultRow = result[i];

			for (int k = 0; k < this->width; k++) {

				const double* objRow = obj.matrix[k];
				double thisElem = this->matrix[i][k];

#pragma loop(no_vector)
				for (int j = 0; j < resultWidth; j++)
					resultRow[j] += thisElem * objRow[j];
			}
		}

		cout << setw(9) << "no vec : " << GetTickCount() - startTime << " milliseconds" << endl;

		return result;
	}

	Matrix sseMul(const Matrix& obj) const {

		Matrix result(this->height, obj.width);
		DWORD startTime = GetTickCount();

		for (int i = 0; i < this->height; i++) {

			double* resultRow = result[i];

			for (int k = 0; k < this->width; k++) {

				const double* objRow = obj.matrix[k];
				__m128d thisElem = _mm_load_pd1(this->matrix[i] + k);

				for (int j = 0; j < obj.width; j += 2)
					_mm_storeu_pd(resultRow + j, _mm_add_pd(_mm_loadu_pd(resultRow + j), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow + j))));
			}
		}

		cout << "vecMul : " << GetTickCount() - startTime << " milliseconds" << endl;

		return result;
	}

	Matrix cacheOptMul(const Matrix& obj) const {
	
		int width = obj.width, height = this->height, eq = this->width;

		Matrix result(height, width);
		DWORD startTime = GetTickCount();

		for (int resHeightOffset = 0; resHeightOffset < height; resHeightOffset +=4 ) {

			for (int innerOffset = 0; innerOffset < eq; innerOffset += 8) {

				for (int resWidthOffset = 0; resWidthOffset < width; resWidthOffset += 8) {

							//Micro core
							for (int i = 0; i < 4; i++) {

								double* resultRow = result[i + resHeightOffset] + resWidthOffset;
								for (int k = 0; k < 8; k++) {

									const double* objRow = obj.matrix[k + innerOffset] + resWidthOffset;
									__m128d thisElem = _mm_load_pd1(this->matrix[i + resHeightOffset] + k + innerOffset);

									_mm_storeu_pd(resultRow    , _mm_add_pd(_mm_loadu_pd(resultRow    ), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow    ))));
									_mm_storeu_pd(resultRow + 2, _mm_add_pd(_mm_loadu_pd(resultRow + 2), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow + 2))));
									_mm_storeu_pd(resultRow + 4, _mm_add_pd(_mm_loadu_pd(resultRow + 4), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow + 4))));
									_mm_storeu_pd(resultRow + 6, _mm_add_pd(_mm_loadu_pd(resultRow + 6), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow + 6))));

									/*for (int j = 0; j < 8; j += 2)
										_mm_storeu_pd(resultRow + j, _mm_add_pd(_mm_loadu_pd(resultRow + j), _mm_mul_pd(thisElem, _mm_loadu_pd(objRow + j))));*/
								}
							}
				}
			}
		}

		cout << " cache : " << GetTickCount() - startTime << " milliseconds" << endl;
		return result;
	}

};

#define SIZE_AMP 500

int main() {

	srand((unsigned)time(NULL));

	Matrix a(SIZE_AMP * 4, SIZE_AMP * 8);
	Matrix b(SIZE_AMP * 8, SIZE_AMP * 8);

	a.fill();
	b.fill();

	//Matrix c = a * b;
	//Matrix e = a.noVecMul(b);
	Matrix x = a.cacheOptMul(b);
	Matrix d = a.sseMul(b);

	if (x == d)
		cout << "Ok";
	else
		cout << "Not OK";

	return 0;
}