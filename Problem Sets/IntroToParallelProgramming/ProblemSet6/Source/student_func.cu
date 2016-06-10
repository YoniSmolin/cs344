//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>

const size_t BLOCK_DIM = 32;
const unsigned char INTERIOR = 1;
const unsigned char BORDER = 2;
const unsigned char EXTERIOR = 0;

const unsigned int JACOBI_ITERATION_COUNT = 800;

//////////////////////////// Forward Declarations /////////////////////////
unsigned char* ComputInteriorMask(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource);
unsigned char* ComputeRegionMap(dim3 blockGrid, dim3 threadGrid, const unsigned char* const d_mask, size_t numColsSource, size_t numRowsSource);
void SeparateToChannels(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource, unsigned char*& d_red, unsigned char*& d_green, unsigned char*& d_blue);
float* InitializeChannelBuffer(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource, const unsigned char* const d_source);
void PerformJacobiLoop(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource,
					   const unsigned char const* d_regionMap, float* d_current, float* d_next, const unsigned char const* d_source, const unsigned char const* d_target);
uchar4* CombineChannels(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource,
						const unsigned char* const d_regionMap, const uchar4* const d_target, const float* const d_red, const float* const d_green, const float* const d_blue);

__device__ float Sum1SubComponent(const unsigned char const* d_regionMap, const float const* d_current, const unsigned char const* d_dest, size_t index);

//////////////////////////// CUDA Kernels /////////////////////////////////
__global__ void ComputeMask(const uchar4* const d_sourceImg, const size_t numRowsSource, const size_t numColsSource, unsigned char* const d_mask)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		uchar4 pixel = d_sourceImg[imageIndex];
		d_mask[imageIndex] = ((pixel.x == 255) && (pixel.y == 255) && (pixel.z == 255)) ? EXTERIOR : INTERIOR;
	}
}

__global__ void ComputeRegionMap(const unsigned char* const d_mask, const size_t numRowsSource, const size_t numColsSource, unsigned char* const d_regionMap)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	bool firstRow = rowIndex == 0;
	bool lastRow = rowIndex == numRowsSource - 1;
	bool firstColumn = columnIndex == 0;
	bool lastColumn = columnIndex == numColsSource - 1;


	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		unsigned char maskValue = d_mask[imageIndex];
		d_regionMap[imageIndex] = maskValue;

		if (maskValue == INTERIOR && !firstRow && !lastRow && !firstColumn && !lastColumn)
		{						
			unsigned char up = d_mask[imageIndex - numColsSource];
			unsigned char down = d_mask[imageIndex + numColsSource];
			unsigned char right = d_mask[imageIndex + 1];
			unsigned char left = d_mask[imageIndex - 1];

			// at least one neighbour is an EXTERIOR pixel
			if (up == EXTERIOR ||down == EXTERIOR || right == EXTERIOR || left == EXTERIOR) d_regionMap[imageIndex] = BORDER;
		}
	}
}

__global__ void SeparateToChannels(const uchar4* const d_sourceImage, size_t numRowsSource, size_t numColsSource, unsigned char* d_red, unsigned char* d_green, unsigned char* d_blue)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		uchar4 pixelValue = d_sourceImage[imageIndex];

		d_red[imageIndex] = pixelValue.x;
		d_green[imageIndex] = pixelValue.y;
		d_blue[imageIndex] = pixelValue.z;
	}
}

__global__ void Copy(const unsigned char* const source, size_t numRowsSource, size_t numColsSource, float* const target)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		target[imageIndex] = (float)source[imageIndex];
	}
}

__global__ void JacobiIteration(size_t numColsSource, size_t numRowsSource, const unsigned char const* d_regionMap, 
								float* d_current, float* d_next, const unsigned char const* d_source, const unsigned char const* d_target)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		
		if (d_regionMap[imageIndex] == INTERIOR)
		{
			size_t up = imageIndex - numColsSource;
			size_t down = imageIndex + numColsSource;
			size_t left = imageIndex - 1;
			size_t right = imageIndex + 1;

			float Sum1 = Sum1SubComponent(d_regionMap, d_current, d_target, left) + 
					     Sum1SubComponent(d_regionMap, d_current, d_target, right) +
						 Sum1SubComponent(d_regionMap, d_current, d_target, up) + 
						 Sum1SubComponent(d_regionMap, d_current, d_target, down);

			float Sum2 = 4 * d_source[imageIndex] - (d_source[up] + d_source[down] + d_source[left] + d_source[right]);

			float newVal = (Sum1 + Sum2) / 4.f;
			d_next[imageIndex] = fminf(255, fmaxf(0, newVal));
		}
	}
}

__device__ float Sum1SubComponent(const unsigned char const* d_regionMap, const float const* d_current, const unsigned char const* d_dest, size_t index)
{
	return d_regionMap[index] == INTERIOR ? d_current[index] : d_dest[index];
}

__global__ void CombineChannels(size_t numColsSource, size_t numRowsSource, const unsigned char* const d_regionMap, 
								const uchar4* const d_target,const float* const d_red, const float* const d_green, const float* const d_blue, uchar4* const d_combined)
{
	size_t columnIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t rowIndex = threadIdx.y + blockDim.y * blockIdx.y;

	if (rowIndex < numRowsSource && columnIndex < numColsSource)
	{
		size_t imageIndex = rowIndex * numColsSource + columnIndex;
		
		if (d_regionMap[imageIndex] == INTERIOR)
		{
			d_combined[imageIndex].x = d_red[imageIndex];
			d_combined[imageIndex].y = d_green[imageIndex];
			d_combined[imageIndex].z = d_blue[imageIndex];
		}
		else
		{
			d_combined[imageIndex] = d_target[imageIndex];
		}
	}
}

//////////////////////////// Main Function ////////////////////////////////////
void your_blend(const uchar4* const h_sourceImg,  const size_t numRowsSource, const size_t numColsSource, const uchar4* const h_destImg, uchar4* const h_blendedImg)
{
	// define grid dimensions
	size_t numPixels = numRowsSource * numColsSource;
	size_t blocksPerRow = (numColsSource + BLOCK_DIM - 1) / BLOCK_DIM;
	size_t blocksPerColumn = (numRowsSource + BLOCK_DIM - 1) / BLOCK_DIM;
	dim3 blockGrid(blocksPerRow, blocksPerColumn, 1);
	dim3 threadGrid(BLOCK_DIM, BLOCK_DIM, 1);

	// move input image to device
	uchar4* d_sourceImage;
	checkCudaErrors(cudaMalloc(&d_sourceImage, numPixels * sizeof(uchar4)));
	checkCudaErrors(cudaMemcpy(d_sourceImage, h_sourceImg, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));

	uchar4* d_targetImage;
	checkCudaErrors(cudaMalloc(&d_targetImage, numPixels * sizeof(uchar4)));
	checkCudaErrors(cudaMemcpy(d_targetImage, h_destImg, numPixels * sizeof(uchar4), cudaMemcpyHostToDevice));

	// 1 - Compute the interior mask 	
	unsigned char* d_mask = ComputInteriorMask(blockGrid, threadGrid, d_sourceImage, numColsSource, numRowsSource);

    // 2 - Compute the interior and border regions of the mask
	unsigned char* d_regionMap = ComputeRegionMap(blockGrid, threadGrid, d_mask, numColsSource, numPixels);

    // 3 - Separate out the incoming image into three channels
	unsigned char *d_redSource, *d_greenSource, *d_blueSource, *d_redTarget, *d_greenTarget, *d_blueTarget;
	SeparateToChannels(blockGrid, threadGrid, d_sourceImage, numColsSource, numRowsSource, d_redSource, d_greenSource, d_blueSource);
	SeparateToChannels(blockGrid, threadGrid, d_targetImage, numColsSource, numRowsSource, d_redTarget, d_greenTarget, d_blueTarget);

    // 4 - Create two float(!) buffers for each color channel
	float* d_redCurrent = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_redSource);
	float* d_redNext = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_redSource);
	float* d_greenCurrent = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_greenSource);
	float* d_greenNext = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_greenSource);
	float* d_blueCurrent = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_blueSource);
	float* d_blueNext = InitializeChannelBuffer(blockGrid, threadGrid, numColsSource, numRowsSource, d_blueSource);

    // 5 - For each color channel perform the Jacobi iteration described above 800 times.
	PerformJacobiLoop(blockGrid, threadGrid, numColsSource, numRowsSource, d_regionMap, d_redCurrent, d_redNext, d_redSource, d_redTarget);
	PerformJacobiLoop(blockGrid, threadGrid, numColsSource, numRowsSource, d_regionMap, d_greenCurrent, d_greenNext, d_greenSource, d_greenTarget);
	PerformJacobiLoop(blockGrid, threadGrid, numColsSource, numRowsSource, d_regionMap, d_blueCurrent, d_blueNext, d_blueSource, d_blueTarget);

	assert(JACOBI_ITERATION_COUNT % 2 == 0);

    // 6 - Create the output image
	uchar4* d_combined = CombineChannels(blockGrid, threadGrid, numColsSource, numRowsSource, d_regionMap, d_targetImage, d_redCurrent, d_greenCurrent, d_blueCurrent);
	checkCudaErrors(cudaMemcpy(h_blendedImg, d_combined, numPixels * sizeof(uchar4), cudaMemcpyDeviceToHost));


	#pragma region cudaFree()
	checkCudaErrors(cudaFree(d_sourceImage));
	checkCudaErrors(cudaFree(d_targetImage));
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_regionMap));
	checkCudaErrors(cudaFree(d_redSource));
	checkCudaErrors(cudaFree(d_greenSource));
	checkCudaErrors(cudaFree(d_blueSource));
	checkCudaErrors(cudaFree(d_redTarget));
	checkCudaErrors(cudaFree(d_greenTarget));
	checkCudaErrors(cudaFree(d_blueTarget));
	checkCudaErrors(cudaFree(d_redCurrent));
	checkCudaErrors(cudaFree(d_greenCurrent));
	checkCudaErrors(cudaFree(d_blueCurrent));
	checkCudaErrors(cudaFree(d_redNext));
	checkCudaErrors(cudaFree(d_greenNext));
	checkCudaErrors(cudaFree(d_blueNext));
	checkCudaErrors(cudaFree(d_combined));
	#pragma endregion
}

//////////////////////////// Helper Functions /////////////////////////////////
unsigned char* ComputInteriorMask(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource)
{
	size_t numPixels = numRowsSource * numColsSource;
	unsigned char* d_mask;
	checkCudaErrors(cudaMalloc(&d_mask, numPixels * sizeof(unsigned char)));

	ComputeMask << <blockGrid, threadGrid >> > (d_sourceImage, numRowsSource, numColsSource, d_mask);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//unsigned char* h_mask = new unsigned char[numPixels];
	//checkCudaErrors(cudaMemcpy(h_mask, d_mask, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//delete[] h_mask;

	return d_mask;
}

unsigned char* ComputeRegionMap(dim3 blockGrid, dim3 threadGrid, const unsigned char* const d_mask, size_t numColsSource, size_t numRowsSource)
{
	size_t numPixels = numRowsSource * numColsSource;
	unsigned char* d_regionMap;
	checkCudaErrors(cudaMalloc(&d_regionMap, numPixels * sizeof(unsigned char)));

	ComputeRegionMap << <blockGrid, threadGrid >> > (d_mask, numRowsSource, numColsSource, d_regionMap);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//unsigned char* h_regionMap = new unsigned char[numPixels];
	//checkCudaErrors(cudaMemcpy(h_regionMap, d_regionMap, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//delete[] h_regionMap;

	return d_regionMap;
} 

void SeparateToChannels(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource, unsigned char*& d_red, unsigned char*& d_green, unsigned char*& d_blue)
{
	size_t numPixels = numRowsSource * numColsSource;
	checkCudaErrors(cudaMalloc(&d_red, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&d_green, numPixels * sizeof(unsigned char)));
	checkCudaErrors(cudaMalloc(&d_blue, numPixels * sizeof(unsigned char)));

	SeparateToChannels << <blockGrid, threadGrid >> > (d_sourceImage, numRowsSource, numColsSource, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//unsigned char* h_red = new unsigned char[numPixels];
	//checkCudaErrors(cudaMemcpy(h_red, d_red, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//delete[] h_red;
}

float* InitializeChannelBuffer(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource, const unsigned char* const d_source)
{
	size_t numPixels = numRowsSource * numColsSource;
	float* d_target;
	checkCudaErrors(cudaMalloc(&d_target, numPixels * sizeof(float)));	

	Copy << <blockGrid, threadGrid >> > (d_source, numRowsSource, numColsSource, d_target);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//float* h_target = new float[numPixels];
	//checkCudaErrors(cudaMemcpy(h_target, d_target, numPixels * sizeof(float), cudaMemcpyDeviceToHost));
	//delete[] h_target;

	return d_target;
}

void PerformJacobiLoop(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource, 
					   const unsigned char const* d_regionMap, float* d_current, float* d_next, const unsigned char const* d_source, const unsigned char const* d_target)
{
	for (unsigned int i = 0; i < JACOBI_ITERATION_COUNT; i++)
	{
		// perform a single iteration
		JacobiIteration << < blockGrid, threadGrid >> > (numColsSource, numRowsSource, d_regionMap, d_current, d_next, d_source, d_target);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//size_t numPixels = numColsSource*numRowsSource;
		//float* h_next = new float[numPixels];
		//checkCudaErrors(cudaMemcpy(h_next, d_next, numPixels * sizeof(float), cudaMemcpyDeviceToHost));
		//delete[] h_next;

		// swap pointers
		float* aux = d_current;
		d_current = d_next;
		d_next = aux;
	}

	// undo final swap
	float* aux = d_current;
	d_current = d_next;
	d_next = aux;
}

uchar4* CombineChannels(dim3 blockGrid, dim3 threadGrid, size_t numColsSource, size_t numRowsSource, 
						const unsigned char* const d_regionMap, const uchar4* const d_target, const float* const d_red, const float* const d_green, const float* const d_blue)
{
	size_t numPixels = numRowsSource * numColsSource;
	uchar4* d_combined;
	checkCudaErrors(cudaMalloc(&d_combined, numPixels * sizeof(uchar4)));

	CombineChannels << <blockGrid, threadGrid >> > (numColsSource, numRowsSource, d_regionMap, d_target, d_red, d_green, d_blue, d_combined);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//unsigned char* h_mask = new unsigned char[numPixels];
	//checkCudaErrors(cudaMemcpy(h_mask, d_mask, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	//delete[] h_mask;

	return d_combined;
}