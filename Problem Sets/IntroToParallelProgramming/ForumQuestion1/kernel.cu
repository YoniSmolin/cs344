
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

int inputSize = 1024;

__global__
void hillisSteeleScanSumSigned(int* d_input, int *d_output, int inputSize)
{
	int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = idx_x + idx_y * (blockDim.x * gridDim.x);

	//Copy the input to d_output, so it's modify-able
	d_output[idx] = d_input[idx];
	__syncthreads();

	for (int i = 1; i <= (inputSize / 2); i <<= 1)
	{
		if (idx >= i)
		{
			int temp = d_output[idx] + d_output[idx - i];
			__syncthreads();
			d_output[idx] = temp;
		}

		//printf("Thread %d finishes step %d\n", idx, i);
		__syncthreads();
	}
}

int main()
{
	int* h_in = (int *)malloc(inputSize * sizeof(int));
	for (int i = 0; i < inputSize; ++i)
		h_in[i] = i + 1;

	int* h_out = (int *)malloc(inputSize * sizeof(int));

	int* d_in;
	int* d_out;

	checkCudaErrors(cudaMalloc(&d_in, sizeof(int) * inputSize));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(int) * inputSize));

	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int)*inputSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_out, 0, sizeof(int)*inputSize));

	dim3 blockSize(32, 32, 1);
	hillisSteeleScanSumSigned << <1, blockSize >> >(d_in, d_out, inputSize);

	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(int) * inputSize, cudaMemcpyDeviceToHost));

	//--------- inclusive sum serially------------------
	int* h_out_s = (int *)malloc(inputSize * sizeof(int));
	h_out_s[0] = h_in[0];
	for (int i = 1; i < inputSize; ++i)
		h_out_s[i] = h_out_s[i - 1] + h_in[i];
	//--------------------------------------------------
	printf("\n");
	printf("h_in \t h_out \t h_out_s\tdiff\n");
	for (int i = 0; i < inputSize; ++i)
	{
		int diff = h_out[i] - h_out_s[i];
		printf("%d \t %d \t %d \t %d \n", h_in[i], h_out[i], h_out_s[i], diff);
	}

    return 0;
}