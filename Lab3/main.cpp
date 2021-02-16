#include <windows.h>
#include <iostream>

using namespace std;

#define B 8
#define KB 1024 * B
#define MB 1024 * KB
#define MAX_ASSOCIATION 20
#define L1_SIZE 32 * KB / sizeof(ull)
#define L2_SIZE 265 * KB / sizeof(ull)
#define L3_SIZE 6 * MB / sizeof(ull)
#define OFFSET 4 * MB / sizeof(ull)
#define TRIES 200

typedef unsigned long long int ull;

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
	
		for (int i = 0; i < OFFSET * MAX_ASSOCIATION; i++)
			stream << obj.array[i] << " ";
	
		return stream;
	}

	void init(size_t size, int assoc) {

		ZeroMemory(array, OFFSET * MAX_ASSOCIATION);

		if (size > L3_SIZE || assoc < 1 || assoc > MAX_ASSOCIATION)
			exit(1);

		if (assoc == 1) {

			for (size_t i = 0; i < size - 1; i++) {

				this->array[i] = i + 1;
			}
			return;
		}

		ull blockSize = size % assoc == 0 ? size / assoc : size / assoc + 1;
		ull block = blockSize;
		ull poz = 0, num = OFFSET;
		for (int i = 0; i < assoc - 1; i++)
		{
			for (int j = 0; j < blockSize; j++)
				array[poz++] = num++;
			poz += OFFSET - blockSize;
			num += OFFSET - blockSize;
			if (i + 1 == size % assoc) array[poz - blockSize--] = 0;
		}
		num = 1;
		for (int j = 0; j < blockSize; j++)
			array[poz++] = num++;
		if (size % assoc == 0) array[poz - 1] = 0;
	}

	DWORD testRead() {

		ull index = 0;
		DWORD startTime = GetTickCount();

		for (int i = 0; i < TRIES; i++) {

			do {
				index = this->array[index];
			} while (index != 0);
		}

		return (GetTickCount() - startTime) / TRIES;
	}


};

int main() {

	TestArray array;

	/*array.init(10, 2);

	cout << array;*/

	for (int i = 1; i <= MAX_ASSOCIATION; i++) {
		
		array.init(L1_SIZE, i);
		cout << i << " : " << array.testRead() << endl;
	
	}

	return 0;
}