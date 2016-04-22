//Udacity HW 4
//Radix Sorting
/* Red Eye Removal
===============

For this assignment we are implementing red eye removal.  This is
accomplished by first creating a score for every pixel that tells us how
likely it is to be a red eye pixel.  We have already done this for you - you
are receiving the scores and need to sort them in ascending order so that we
know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

*/

#include "utils.h"
#include <thrust/host_vector.h>

//////////////////////////////// Forward Declerations ///////////////////
__device__ void ComputeBinaryPredicate(const unsigned int* const inputValues, unsigned int* const outputPredicates, const unsigned int filter, const bool parity, const size_t numElems);
__device__ void HillisSteele_InclusiveSum_BlockScan(const unsigned int* const sourceArray, unsigned int* const targetArray, const size_t numElems);

//////////////////////////////// Constants //////////////////////////////
const unsigned int THREADS_PER_BLOCK = 512;
const bool PARITY_EVEN = true;
const bool PARITY_ODD = false;

/////////////////////////////// Device Variables ////////////////////////
__device__ unsigned int d_globalOffset;

/////////////////////////////// CUDA Kernels ////////////////////////////
__global__ void BlockScan_AccordingToPredicate(const unsigned int* const inputValues, unsigned int* const outputPredicates, unsigned int* const outputScannedArray, const unsigned int filter, 
											   const bool parity, const size_t numElems)
{
	ComputeBinaryPredicate(inputValues, outputPredicates, filter, parity, numElems);
	__syncthreads();
	HillisSteele_InclusiveSum_BlockScan(outputPredicates, outputScannedArray, numElems);
}

__global__ void ComputeBlockScanOffsets(const unsigned int* const indices, unsigned int* const offsets, const unsigned int threadsPerBlock, const bool parity, const size_t numElems)
{
	auto threadId = threadIdx.x;
	auto blockSize = blockDim.x;

	offsets[threadId] = indices[threadsPerBlock-1 + (threadsPerBlock * threadId)];
	HillisSteele_InclusiveSum_BlockScan(offsets, offsets, blockSize);

	if (parity) d_globalOffset = offsets[blockSize-1] + indices[numElems-1];
}

__global__ void Scatter_AccordingToScan(const unsigned int* const inputVals, const unsigned int* const inputPos, unsigned int* const outputVals, unsigned int* const outputPos,
									    const unsigned int* const targetIndices, const unsigned int* const targetOffsets, const unsigned int* const predicates, const bool parity, 
										const size_t numElems)
{
	auto threadId = threadIdx.x;
	auto blockId = blockIdx.x;
	auto globalIndex = threadId + blockDim.x * blockId;

	if (globalIndex < numElems && predicates[globalIndex] == 1)
	{
		auto globalOffset = parity ? 0 : d_globalOffset;
		auto indexInTargetArray = globalOffset + (targetIndices[globalIndex] - 1) + ((blockId == 0) ? 0 : targetOffsets[blockId-1]); // (-1) because of the inclusive sum scan
		outputVals[indexInTargetArray] = inputVals[globalIndex];
		outputPos[indexInTargetArray] = inputPos[globalIndex];
	}
}

/////////////////////////////// CUDA Device Methods /////////////////////
__device__ void ComputeBinaryPredicate(const unsigned int* const inputValues, unsigned int* const outputPredicates, const unsigned int filter, const bool parity, const size_t numElems)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElems)
	{
		auto inputValue = inputValues[index];
		auto inputValueFiltered = inputValue & filter;
		outputPredicates[index] = parity ? (filter != inputValueFiltered) : (filter == inputValueFiltered);
	}
}

__device__ void HillisSteele_InclusiveSum_BlockScan(const unsigned int* const sourceArray, unsigned int* const targetArray, const size_t numElems)
{
	auto blockSize = blockDim.x;
	auto threadId = threadIdx.x;
	auto offset = blockIdx.x * blockSize;
	auto globalIndex = offset + threadId;

	if (globalIndex < numElems)
	{
		targetArray[globalIndex] = sourceArray[globalIndex];
		__syncthreads();
		for (auto leaves = 1; leaves < blockSize; leaves *= 2)
		{
			if (threadId >= leaves)
			{
				auto right = targetArray[globalIndex];
				auto left = targetArray[globalIndex - leaves];
				__syncthreads();
				targetArray[globalIndex] = right + left;
			}
			__syncthreads();
		}
	}
}

/////////////////////////////// Sort Function ///////////////////////////
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
	unsigned int bitsInUnsignedInt = sizeof(unsigned int) * 8;

	unsigned int numberOfBlocks = (numElems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	unsigned int* d_predicates;
	unsigned int* d_targetIndices;
	unsigned int* d_targetOffsets;
	checkCudaErrors(cudaMalloc(&d_predicates, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&d_targetIndices, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMalloc(&d_targetOffsets, sizeof(unsigned int) * (numberOfBlocks - 1)));

	unsigned int *sourceVals, *targetVals, *sourcePos, *targetPos;

	for (unsigned int i = 0; i < bitsInUnsignedInt; i++)
	{
		unsigned int filter = 1 << i;
		
		if (i % 2 == 0)
		{
			sourceVals = d_inputVals;
			sourcePos = d_inputPos;
			targetVals = d_outputVals;
			targetPos = d_outputPos;
		}
		else
		{
			sourceVals = d_outputVals;
			sourcePos = d_outputPos;
			targetVals = d_inputVals;
			targetPos = d_inputPos;
		}

		// Handle zeros
		BlockScan_AccordingToPredicate << < numberOfBlocks, THREADS_PER_BLOCK >> > (sourceVals, d_predicates, d_targetIndices, filter, PARITY_EVEN, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		assert(numberOfBlocks > 0);
		ComputeBlockScanOffsets << < 1, numberOfBlocks - 1 >> > (d_targetIndices, d_targetOffsets, THREADS_PER_BLOCK, PARITY_EVEN, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		Scatter_AccordingToScan << < numberOfBlocks, THREADS_PER_BLOCK >> > (sourceVals, sourcePos, targetVals, targetPos, d_targetIndices, d_targetOffsets, d_predicates, PARITY_EVEN, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// Handle ones
		BlockScan_AccordingToPredicate << < numberOfBlocks, THREADS_PER_BLOCK >> > (sourceVals, d_predicates, d_targetIndices, filter, PARITY_ODD, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		assert(numberOfBlocks > 0);
		ComputeBlockScanOffsets << < 1, numberOfBlocks - 1 >> > (d_targetIndices, d_targetOffsets, THREADS_PER_BLOCK, PARITY_ODD, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		Scatter_AccordingToScan << < numberOfBlocks, THREADS_PER_BLOCK >> > (sourceVals, sourcePos, targetVals, targetPos, d_targetIndices, d_targetOffsets, d_predicates, PARITY_ODD, numElems);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}

	assert(bitsInUnsignedInt % 2 == 0);
	// after 32 iterations, the sorted array is in the input array
	checkCudaErrors(cudaMemcpy(targetVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(targetPos, d_outputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));

	//unsigned int* h_predicates = new unsigned int[numElems];
	//checkCudaErrors(cudaMemcpy(h_predicates, d_predicates, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	//unsigned int* h_inputValues = new unsigned int[numElems];
	//checkCudaErrors(cudaMemcpy(h_inputValues, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	//unsigned int* h_targetIndices = new unsigned int[numElems];
	//checkCudaErrors(cudaMemcpy(h_targetIndices, d_targetIndices, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	//unsigned int* h_targetOffsets = new unsigned int[numberOfBlocks - 1];
	//checkCudaErrors(cudaMemcpy(h_targetOffsets, d_targetOffsets, sizeof(unsigned int) * (numberOfBlocks-1), cudaMemcpyDeviceToHost));
	//unsigned int* h_outputValues = new unsigned int[numElems];
	//checkCudaErrors(cudaMemcpy(h_outputValues, d_outputVals, sizeof(unsigned int) * (numElems), cudaMemcpyDeviceToHost));
	//unsigned int* h_outputPositions = new unsigned int[numElems];
	//checkCudaErrors(cudaMemcpy(h_outputPositions, d_outputPos, sizeof(unsigned int) * (numElems), cudaMemcpyDeviceToHost));

	//delete[] h_predicates;
	//delete[] h_inputValues;
	//delete[] h_targetIndices;
	//delete[] h_targetOffsets;
	//delete[] h_outputValues;
	//delete[] h_outputPositions;

	checkCudaErrors(cudaFree(d_predicates));
	checkCudaErrors(cudaFree(d_targetOffsets));
	checkCudaErrors(cudaFree(d_targetIndices));
}
