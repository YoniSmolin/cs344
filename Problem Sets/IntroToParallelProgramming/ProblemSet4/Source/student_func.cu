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

//////////////////////////////// Constants //////////////////////////////
const unsigned int THREADS_PER_BLOCK = 512;
const bool PARITY_EVEN = true;
const bool PARITY_ODD = false;

/////////////////////////////// CUDA Kernels ////////////////////////////
__global__ void ComputeBinaryPredicate(const unsigned int* const inputValues, unsigned int* const outputPredicates, const unsigned int filter, const bool parity, const size_t numElems)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numElems)
	{
		auto inputValue = inputValues[index];
		auto inputValueFiltered = inputValue & filter;
		outputPredicates[index] = parity ? (filter != inputValueFiltered) : (filter == inputValueFiltered);
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
	checkCudaErrors(cudaMalloc(&d_predicates, sizeof(unsigned int) * numElems));

	for (unsigned int i = 0; i < bitsInUnsignedInt; i++)
	{
		unsigned int filter = 1 << i;
		
		// gather even values
		ComputeBinaryPredicate << < numberOfBlocks, THREADS_PER_BLOCK >> > (d_inputVals, d_predicates, filter, PARITY_EVEN, numElems);

		// gather odd values
		break;
	}

	unsigned int* h_predicates = new unsigned int[numElems];
	checkCudaErrors(cudaMemcpy(h_predicates, d_predicates, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	unsigned int* h_inputValues = new unsigned int[numElems];
	checkCudaErrors(cudaMemcpy(h_inputValues, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaFree(d_predicates));
	delete[] h_predicates;
}
