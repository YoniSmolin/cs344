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

__global__
void Serial_SharedMem(const unsigned int* const vals, unsigned int* const histo, const unsigned int numBins, const unsigned int numElems)
{	
	extern __shared__ unsigned int histogram[];	

	for (unsigned int i = 0; i < numBins; i++) histogram[i] = 0;

	for (unsigned int i = 0; i < numElems; i++) histogram[vals[i]]++;

	for (unsigned int i = 0; i < numBins; i++) histo[i] = histogram[i];
}

void computeHistogram(const unsigned int* const d_vals, unsigned int* const d_histo, const unsigned int numBins, const unsigned int numElems)
{
	Serial_SharedMem << <1, 1, numBins * sizeof(unsigned int) >> > (d_vals, d_histo, numBins, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
