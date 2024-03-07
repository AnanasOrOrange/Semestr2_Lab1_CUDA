#include "cuda_runtime.h"					
#include "device_launch_parameters.h"		
#include <iostream>
#include <chrono>

#define OUT_MATRIX
#define BLOCK_SIZE 16

static int N = 4, M = 3, P = 2;

#define CHECK_ERR()														
void check() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		system("pause");
		exit(1);
	}
}

void fillByRandom(int** matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i][j] = rand() % 10;
		}
	}
}

#ifdef OUT_MATRIX
void printMatrix(int** matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
#endif

void matrixMultiplicationCPU(int** A, int** B, int** C, int rowsA, int colsA, int rowsB, int colsB) {
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < colsB; j++) {
			C[i][j] = 0;
			for (int k = 0; k < colsA; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

__global__
void matrixMultiplicationGPU(int* A, int* B, int* C, int rowsA, int colsA, int rowsB, int colsB) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// (N x M)  X  (M x P)  = (N x P) 
	int n = rowsA;
	int m = colsA;
	int p = colsB;

	if (row < n && col < p) {
		int sum = 0;
		for (int i = 0; i < m; i++) {
			sum += A[row * m + i] * B[i * p + col];
		}
		C[row * p + col] = sum;
	}

}

__global__
void tr_matrixMultiplicationGPU(int* A, int* B, int* C, int rowsA, int colsA, int rowsB, int colsB) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// (N x M)  X  (M x P)  = (N x P) 
	int n = rowsA;
	int m = colsA;
	int p = colsB;

	if (row < n && col < p) {
		int sum = 0;
		for (int i = 0; i < m; i++) {
			sum += A[row * m + i] * B[col * m + i];
		}
		C[row * p + col] = sum;
	}
}

__global__ 
void sh_matrixMultiplicationGPU(int* A, int* B, int* C, int m, int n, int k) {
	__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;

	float result = 0;

	for (int i = 0; i < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
		if (row < m && i * BLOCK_SIZE + tx < n) {
			sA[ty][tx] = A[row * n + i * BLOCK_SIZE + tx];
		}
		else {
			sA[ty][tx] = 0;
		}

		if (i * BLOCK_SIZE + ty < n && col < k) {
			sB[ty][tx] = B[(i * BLOCK_SIZE + ty) * k + col];
		}
		else {
			sB[ty][tx] = 0;
		}

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; ++j) {
			result += sA[ty][j] * sB[j][tx];
		}

		__syncthreads();
	}

	if (row < m && col < k) {
		C[row * k + col] = result;
	}
}



void copyMatrix_2dTo1d(int** matrix_2d, int rows, int cols, int* matrix_1d) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix_1d[i * cols + j] = matrix_2d[i][j];
		}
	}
}

void copyMatrix_1dTo2d(int** matrix_2d, int rows, int cols, int* matrix_1d) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix_2d[i][j] = matrix_1d[i * cols + j];
		}
	}
}

bool equelMatrix(int** A, int** B, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (A[i][j] != B[i][j]) {
				return false;
			}
		}
	}
	return true;
}

void transpose(int* a, int* b, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			b[j * rows + i] = a[i * cols + j];
	}
}

int main()
{
	//               
	int rowsA = N, colsA = M;
	int rowsB = colsA, colsB = P;

	std::cout << "(N x M)  X  (M x P)  = (N x P) " << std::endl;
	std::cout << "N = " << rowsA << " M = " << colsA << " P = " << colsB << std::endl;
	std::cout << std::endl;

	int** A = new int* [rowsA];
	int** B = new int* [rowsB];

	for (int i = 0; i < rowsA; i++) {
		A[i] = new int[colsA];
	}
	for (int i = 0; i < rowsB; i++) {
		B[i] = new int[colsB];
	}

	fillByRandom(A, rowsA, colsA);
	fillByRandom(B, rowsB, colsB);

#ifdef OUT_MATRIX
	std::cout << "Matrix A:" << std::endl;
	printMatrix(A, rowsA, colsA);

	std::cout << "Matrix B:" << std::endl;
	printMatrix(B, rowsB, colsB);
#endif

	/////////////////////////////////////////

	//               CPU

	int rowsC = rowsA, colsC = colsB;
	int** C = new int* [rowsC];
	for (int i = 0; i < rowsC; i++) {
		C[i] = new int[colsC];
	}

	//          A x B = C

	auto startCPU = std::chrono::steady_clock::now();
	matrixMultiplicationCPU(A, B, C, rowsA, colsA, rowsB, colsB);
	auto endCPU = std::chrono::steady_clock::now();

	//                                       xD                    
	std::cout << "CPU matrix multiplication time = " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << " millisec, (1e-3 sec)" << std::endl;

#ifdef OUT_MATRIX
	std::cout << "A x B = Matrix C:" << std::endl;
	printMatrix(C, rowsC, colsC);
#endif

	//////////////////////////////////////////

	//            only CUDA

	cudaEvent_t startGPU, stopGPU;

	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	cudaEventRecord(startGPU);

	int* hostA = new int[rowsA * colsA];
	int* hostB = new int[rowsB * colsB];
	int* hostC = new int[rowsC * colsC];

	int* devA, * devB, * devC;

	cudaMalloc((void**)&devA, rowsA * colsA * sizeof(int)); CHECK_ERR();
	cudaMalloc((void**)&devB, rowsB * colsB * sizeof(int)); CHECK_ERR();
	cudaMalloc((void**)&devC, rowsC * colsC * sizeof(int)); CHECK_ERR();

	copyMatrix_2dTo1d(A, rowsA, colsA, hostA); CHECK_ERR();
	copyMatrix_2dTo1d(B, rowsB, colsB, hostB); CHECK_ERR();

	cudaMemcpy(devA, hostA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, hostB, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((colsC + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowsC + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, rowsA, colsA, rowsB, colsB); CHECK_ERR();

	cudaMemcpy(hostC, devC, rowsC * colsC * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();
	copyMatrix_1dTo2d(C, rowsC, colsC, hostC);

	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	float timeCUDA = 0;
	cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

	std::cout << "GPU matrix multiplication time = " << timeCUDA << " millisec, (1e-3 sec)" << std::endl;
#ifdef OUT_MATRIX
	std::cout << "A x B = Matrix C:" << std::endl;
	printMatrix(C, rowsC, colsC);
#endif

	//////////////////////////////////////////

	//              CUDA + Transponse         

	cudaEventRecord(startGPU);

	copyMatrix_2dTo1d(A, rowsA, colsA, hostA); CHECK_ERR();
	copyMatrix_2dTo1d(B, rowsB, colsB, hostB); CHECK_ERR();

	int* tr_hostB = new int[rowsB * colsB];
	transpose(hostB, tr_hostB, rowsB, colsB);

	cudaMemcpy(devA, hostA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, tr_hostB, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	tr_matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, rowsA, colsA, rowsB, colsB); CHECK_ERR();

	cudaMemcpy(hostC, devC, rowsC * colsC * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();
	copyMatrix_1dTo2d(C, rowsC, colsC, hostC);

	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

	std::cout << "GPU matrix multiplication time with Transpons = " << timeCUDA << " millisec, (1e-3 sec)" << std::endl;
#ifdef OUT_MATRIX
	std::cout << "A x B = Matrix C:" << std::endl;
	printMatrix(C, rowsC, colsC);
#endif


	//////////////////////////////////////////

	//              CUDA + shared

	cudaEventRecord(startGPU);

	copyMatrix_2dTo1d(A, rowsA, colsA, hostA); CHECK_ERR();
	copyMatrix_2dTo1d(B, rowsB, colsB, hostB); CHECK_ERR();

	cudaMemcpy(devA, hostA, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, hostB, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	for (int i = 0; i < rowsC * colsC; i++) {
		hostC[i] = 0;
	}

	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	sh_matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, rowsA, colsA, colsB);

	cudaMemcpy(hostC, devC, rowsC * colsC * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();
	copyMatrix_1dTo2d(C, rowsC, colsC, hostC);

	cudaEventRecord(stopGPU);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

	std::cout << "GPU matrix multiplication time with Shared memory = " << timeCUDA << " millisec, (1e-3 sec)" << std::endl;
#ifdef OUT_MATRIX
	std::cout << "A x B = Matrix C:" << std::endl;
	printMatrix(C, rowsC, colsC);
#endif

	// cleaning memory         

	cudaEventDestroy(startGPU);
	cudaEventDestroy(stopGPU);

	for (int i = 0; i < rowsA; i++) {
		delete[] A[i];
	}
	for (int i = 0; i < rowsB; i++) {
		delete[] B[i];
	}
	for (int i = 0; i < rowsC; i++) {
		delete[] C[i];
	}

	delete[] A, B, C;
	delete[] hostA, hostB, hostC, tr_hostB;

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	return 0;
}
