﻿/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <limits>

// forward declarations
void FindMinMax(const float* const d_logLuminance, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols);
void ComputeHistogram(const float* const d_logLuminance, unsigned int* d_histogram, float min_logLum, float lumRange, const size_t numRows, const size_t numCols, const size_t numBins);
__device__ void ArrayMinMax(float* array, int arraySize, bool computeMax);

__global__ void MinMaxKernel(const float* const inputArray, float* outputArray, bool computeMax)
{
	// variables
	extern __shared__ float shared[];

	int arrayStride = blockDim.x * 2;
	int arrayStartIndex = blockIdx.x * arrayStride;
	int threadId = threadIdx.x;

	// global → local
	shared[threadId] = inputArray[arrayStartIndex + threadId];
	shared[arrayStride / 2 + threadId] = inputArray[arrayStartIndex + arrayStride / 2 + threadId];

	__syncthreads();

	ArrayMinMax(shared, arrayStride, computeMax);

	outputArray[blockIdx.x] = shared[0];
}

__device__ void ArrayMinMax(float* array, int arraySize, bool computeMax)
{
	int threadId = threadIdx.x;

	for (int activeThreads = arraySize / 2; activeThreads > 0; activeThreads /= 2)
	{
		if (threadId < activeThreads)
		{
			auto firstValue = array[threadId];
			auto secondValue = array[activeThreads + threadId];
			if (computeMax)
				array[threadId] = firstValue > secondValue ? firstValue : secondValue;
			else
				array[threadId] = firstValue < secondValue ? firstValue : secondValue;
			__syncthreads();
		}
	}

}

__global__ void FillHistogramTable(const float* const d_logLuminance, unsigned int* d_histogtam, float min_logLum, float lumRange, size_t numBins)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	float luminance = d_logLuminance[index];

	unsigned int binIndex = (unsigned int)(numBins * (luminance - min_logLum) / lumRange);

	atomicAdd(&d_histogtam[binIndex], 1);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	// step 0 - check assumptions

	assert(numCols % 2 == 0 && numRows % 2 == 0);

	// step 1 - minimum and maximum
	FindMinMax(d_logLuminance, min_logLum, max_logLum, numRows, numCols);

	// step 2 - compute histogram
	size_t histogramSize = (size_t)(numBins * sizeof(unsigned int));

	unsigned int* d_histogram;
	checkCudaErrors(cudaMalloc(&d_histogram, histogramSize));

	ComputeHistogram(d_logLuminance, d_histogram, min_logLum, max_logLum - min_logLum, numRows, numCols, numBins);

	checkCudaErrors(cudaFree(d_histogram));
}

void FindMinMax(const float* const d_logLuminance, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols)
{
	size_t rowSize = (size_t)(numCols * sizeof(float));
	size_t columnSize = (size_t)(numRows * sizeof(float));

	float* d_localMaxima;
	float* d_globalMaximum;
	float* d_localMinima;
	float* d_globalMinimum;

	checkCudaErrors(cudaMalloc(&d_localMaxima, columnSize));
	checkCudaErrors(cudaMalloc(&d_globalMaximum, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_localMinima, columnSize));
	checkCudaErrors(cudaMalloc(&d_globalMinimum, sizeof(float)));

	MinMaxKernel << < numRows, numCols / 2, rowSize >> > (d_logLuminance, d_localMaxima, true);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	MinMaxKernel << < 1, numRows / 2, columnSize >> > (d_localMaxima, d_globalMaximum, true);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	MinMaxKernel << < numRows, numCols / 2, rowSize >> > (d_logLuminance, d_localMinima, false);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	MinMaxKernel << < 1, numRows / 2, columnSize >> > (d_localMinima, d_globalMinimum, false);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&max_logLum, d_globalMaximum, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(&min_logLum, d_globalMinimum, sizeof(float), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_localMaxima));
	checkCudaErrors(cudaFree(d_globalMaximum));
	checkCudaErrors(cudaFree(d_localMinima));
	checkCudaErrors(cudaFree(d_globalMinimum));
}

void ComputeHistogram(const float* const d_logLuminance, unsigned int* d_histogram, float min_logLum, float lumRange, const size_t numRows, const size_t numCols, const size_t numBins)
{
	size_t histogramSize = (size_t)(numBins * sizeof(unsigned int));

	checkCudaErrors(cudaMemset(d_histogram, 0, histogramSize));

	FillHistogramTable << < numRows, numCols >> > (d_logLuminance, d_histogram, min_logLum, lumRange, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}