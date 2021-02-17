#include <windows.h>
#include <iostream>
#include <iomanip>
#include <intrin.h>

#pragma intrinsic(__rdtsc)

#define KB 1024
#define MB 1024 * KB
#define CACHE_SIZE 6 * MB / sizeof(ull)
#define OFFSET 4 * MB / sizeof(ull)
#define MAX_ASSOCIATION 20
#define TRIES 100

typedef unsigned long long int ull;

using namespace std;

class TestArray {

private:

	ull* array;

public:

	TestArray(){
	
		array = new ull[OFFSET * MAX_ASSOCIATION];
	}

	~TestArray() {
	
		delete[] array;
	}

	friend ostream& operator <<(ostream& stream, const TestArray& obj) {
	
		size_t i = 0;
		do{
			stream << obj.array[i] << " "; 
			i = obj.array[i];
		} while (i);
	
		return stream;
	} 

	//Формирование нужной структуры памяти
	void init(size_t size, int assoc) {

		ZeroMemory(array, OFFSET * MAX_ASSOCIATION);

		if (assoc == 1) {

			for (size_t i = 0; i < size - 1; i++) {

				this->array[i] = i + 1;
			}
			return;
		}

		size_t blockSize = size % assoc == 0 ? size / assoc : size / assoc + 1;
		size_t currentOffset = 0;

		for (size_t i = 0; i < assoc - 1; i++) {

			for (size_t j = 0; j < blockSize; j++)
				array[currentOffset + j] = currentOffset + OFFSET + j;

			currentOffset += OFFSET;
		}

		for (size_t i = 0; i < blockSize; i++)
			array[currentOffset + i] = i + 1;
	}

	//Замер числа тактов TRIES обходов
	ull testRead() {

		ull index = 0;
		ull startTicks = __rdtsc();

		for (int i = 0; i < TRIES; i++) {

			do {
				index = this->array[index];
			} while (index);
		}

		return (__rdtsc() - startTicks)/TRIES;
	}


};

int main() {

	TestArray array;

	for (int i = 1; i <= MAX_ASSOCIATION; i++) {
		
		array.init(CACHE_SIZE, i);
		cout << setw(2) << i << " : " << setw(8) << array.testRead() << " ticks" << endl;
	}

	return 0;
}