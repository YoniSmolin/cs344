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

//////////////////////////// Forward Declarations /////////////////////////
unsigned char* ComputInteriorMask(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource);
unsigned char* ComputeRegionMap(dim3 blockGrid, dim3 threadGrid, const unsigned char* const d_mask, size_t numColsSource, size_t numRowsSource);
void SeparateToChannels(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numRowsSource, unsigned char*& d_red, unsigned char*& d_green, unsigned char*& d_blue);

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

			if ((up & down & right & left) == 0) d_regionMap[imageIndex] = BORDER;
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

	// 1 - Compute the interior mask 	
	unsigned char* d_mask = ComputInteriorMask(blockGrid, threadGrid, d_sourceImage, numColsSource, numRowsSource);

    // 2 - Compute the interior and border regions of the mask
	unsigned char* d_regionMap = ComputeRegionMap(blockGrid, threadGrid, d_mask, numColsSource, numPixels);

    // 3 - Separate out the incoming image into three channels
	unsigned char *d_red, *d_green, *d_blue;
	SeparateToChannels(blockGrid, threadGrid, d_sourceImage, numColsSource, numRowsSource, d_red, d_green, d_blue);

    // Create two float(!) buffers for each color channel

/*     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_regionMap));
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
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