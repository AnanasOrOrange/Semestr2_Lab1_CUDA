#include "cuda_runtime.h"					
#include "device_launch_parameters.h"		
#include <iostream>
#include <chrono>

#define OUT_MATRIX
#define BLOCK_SIZE	16

static int M = 4, N = 3, P = 2;

#define CHECK_ERR()														
void check() {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d: %s\M", __FILE__, __LINE__, cudaGetErrorString(err));
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

void matrixMultiplicationCPU(int** A, int** B, int** C) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < P; j++) {
			C[i][j] = 0;
			for (int k = 0; k < N; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

__global__
void matrixMultiplicationGPU(int* A, int* B, int* C, int M, int N, int P) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// (M x N)  X  (N x P)  = (M x P) 

	if (row < M && col < P) {
		int sum = 0;
		for (int i = 0; i < N; i++) {
			sum += A[row * N + i] * B[i * P + col];
		}
		C[row * P + col] = sum;
	}

}

__global__
void tr_matrixMultiplicationGPU(int* A, int* B, int* C, int M, int N, int P) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// (M x N)  X  (N x P)  = (M x P) 

	if (row < M && col < P) {
		int sum = 0;
		for (int i = 0; i < N; i++) {
			// B[N x P] -> Transpose -> B[P x N]
			sum += A[row * N + i] * B[col * N + i];
		}
		C[row * P + col] = sum;
	}
}

__global__ 
void sh_matrixMultiplicationGPU(int* A, int* B, int* C, int M, int N, int P) {

	__shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];

	//	вместо того, чтобы постоянно обращаться к глобальной памяти
	//	мы просто используем shared память размером с блок

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// (M x N)  X  (N x P)  = (M x P) 

	float sum = 0.0f;

	// идем по количеству блоков, нужных для умножения
	for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
		if (row < M && i * BLOCK_SIZE + threadIdx.x < N) {
			shA[threadIdx.y][threadIdx.x] = A[row * N + i * BLOCK_SIZE + threadIdx.x];
		}
		else {
			shA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (col < P && i * BLOCK_SIZE + threadIdx.y < N) {
			shB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * P + col];
		}
		else {
			shB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++) {
			sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < M && col < P) {
		C[row * P + col] = sum;
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

	std::cout << "(M x N)  X  (N x P)  = (M x P) " << std::endl;
	std::cout << "M = " << M << " N = " << N << " P = " << P << std::endl;
	std::cout << std::endl;

	int** A = new int* [M];
	int** B = new int* [N];

	for (int i = 0; i < M; i++) {
		A[i] = new int[N];
	}
	for (int i = 0; i < N; i++) {
		B[i] = new int[P];
	}

	fillByRandom(A, M, N);
	fillByRandom(B, N, P);

#ifdef OUT_MATRIX
	std::cout << "Matrix A:" << std::endl;
	printMatrix(A, M, N);

	std::cout << "Matrix B:" << std::endl;
	printMatrix(B, N, P);
#endif

	/////////////////////////////////////////

	//               CPU

	int rowsC = M, colsC = P;
	int** C = new int* [rowsC];
	for (int i = 0; i < rowsC; i++) {
		C[i] = new int[colsC];
	}

	//          A x B = C

	auto startCPU = std::chrono::steady_clock::now();
	matrixMultiplicationCPU(A, B, C);
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

	int* hostA = new int[M * N];
	int* hostB = new int[N * P];
	int* hostC = new int[rowsC * colsC];

	int* devA, * devB, * devC;

	cudaMalloc((void**)&devA, M * N * sizeof(int)); CHECK_ERR();
	cudaMalloc((void**)&devB, N * P * sizeof(int)); CHECK_ERR();
	cudaMalloc((void**)&devC, rowsC * colsC * sizeof(int)); CHECK_ERR();

	copyMatrix_2dTo1d(A, M, N, hostA); 
	copyMatrix_2dTo1d(B, N, P, hostB);

	cudaMemcpy(devA, hostA, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, hostB, N * P * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((rowsC + BLOCK_SIZE - 1) / BLOCK_SIZE, (colsC + BLOCK_SIZE - 1) / BLOCK_SIZE);

	matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, M, N, P); CHECK_ERR();

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

	copyMatrix_2dTo1d(A, M, N, hostA); 
	copyMatrix_2dTo1d(B, N, P, hostB); 

	int* tr_hostB = new int[N * P];
	transpose(hostB, tr_hostB, N, P);

	cudaMemcpy(devA, hostA, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, tr_hostB, N * P * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	tr_matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, M, N, P); CHECK_ERR();

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

	copyMatrix_2dTo1d(A, M, N, hostA);
	copyMatrix_2dTo1d(B, N, P, hostB); 

	cudaMemcpy(devA, hostA, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
	cudaMemcpy(devB, hostB, N * P * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	cudaMemcpy(devC, hostC, rowsC * colsC * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();

	sh_matrixMultiplicationGPU << < dimGrid, dimBlock >> > (devA, devB, devC, M, N, P);

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

	for (int i = 0; i < M; i++) {
		delete[] A[i];
	}
	for (int i = 0; i < N; i++) {
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
