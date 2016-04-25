/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

/////////////////////////////// Forward Declarations ////////////////////
__device__ void Reduce(unsigned int* const array);

/////////////////////////////// CUDA Kernels ////////////////////////////
__global__ void Serial_SharedMem(const unsigned int* const vals, unsigned int* const histo, const unsigned int numBins, const unsigned int numElems)
{	
	extern __shared__ unsigned int histogram[];	

	for (unsigned int i = 0; i < numBins; i++) histogram[i] = 0;

	for (unsigned int i = 0; i < numElems; i++) histogram[vals[i]]++;

	for (unsigned int i = 0; i < numBins; i++) histo[i] = histogram[i];
}

__global__ void SingleBlockPerBin_SharedMemReduction(const unsigned int* const vals, unsigned int* const histo, const unsigned int numBins, const unsigned int numElems)
{
	extern __shared__ unsigned int binPredicates[];

	auto blockSize = blockDim.x;
	auto binId = blockIdx.x;
	auto threadId = threadIdx.x;
	auto numIterations = numElems / blockSize;
	unsigned int totalSum = 0;

	for (size_t i = 0; i < numIterations; i++)
	{
		auto index = threadId + i * blockSize;
		binPredicates[threadId] = vals[index] == binId ? 1 : 0;
		__syncthreads();
		Reduce(binPredicates);
		totalSum += binPredicates[0];
	}

	if (threadId == 0) histo[binId] = totalSum;
}

/////////////////////////////// CUDA Device Methods /////////////////////
__device__ void Reduce(unsigned int* const array)
{
	auto blockSize = blockDim.x;
	auto threadId = threadIdx.x;

	for (size_t participatingThreads = blockSize / 2; participatingThreads > 0; participatingThreads /= 2)
	{
		if (threadId < participatingThreads)
		{
			auto right = array[threadId + participatingThreads];
			auto left = array[threadId];
			array[threadId] = left + right;
		}
		__syncthreads();
	}
}

void computeHistogram(const unsigned int* const d_vals, unsigned int* const d_histo, const unsigned int numBins, const unsigned int numElems)
{
	// 1st attempt - single thread with shared memory
	//Serial_SharedMem << <1, 1, numBins * sizeof(unsigned int) >> > (d_vals, d_histo, numBins, numElems);
	//cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// 2nd attemept - 1024 blocks X 1024 threads - each block responsible of a different bin, with reduction in shared memory
	assert(1024 * (numElems / 1024) == numElems);
	SingleBlockPerBin_SharedMemReduction << < numBins, 1024, numBins * sizeof(unsigned int) >> > (d_vals, d_histo, numBins, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
