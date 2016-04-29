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
const unsigned int INTERIOR = 1;
const unsigned int BORDER = 2;
const unsigned int EXTERIOR = 0;

//////////////////////////// Forward Declarations /////////////////////////
bool* ComputInteriorMask(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numPixels);

//////////////////////////// CUDA Kernels /////////////////////////////////
__global__ void ComputeMask(const uchar4* const d_sourceImg, const size_t numColsSource, bool* const d_mask)
{
	size_t rowIndex = threadIdx.x + blockDim.x * blockIdx.x;
	size_t columnIndex = threadIdx.y + blockDim.y * blockIdx.y;

	size_t imageIndex = rowIndex * numColsSource + columnIndex;

	uchar4 pixel = d_sourceImg[imageIndex];

	d_mask[imageIndex] = !((pixel.x == 255) && (pixel.y == 255) && (pixel.z == 255));
}

//////////////////////////// Host Code ////////////////////////////////////
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
	bool* d_mask = ComputInteriorMask(blockGrid, threadGrid, d_sourceImage, numColsSource, numPixels);

    // 2 - Compute the interior and border regions of the mask


  /*   3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
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
}

bool* ComputInteriorMask(dim3 blockGrid, dim3 threadGrid, const uchar4* const d_sourceImage, size_t numColsSource, size_t numPixels)
{
	bool* d_mask;
	checkCudaErrors(cudaMalloc(&d_mask, numPixels * sizeof(bool)));

	ComputeMask << <blockGrid, threadGrid >> > (d_sourceImage, numColsSource, d_mask);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	bool* h_mask = new bool[numPixels];
	checkCudaErrors(cudaMemcpy(h_mask, d_mask, numPixels * sizeof(bool), cudaMemcpyDeviceToHost));
	delete[] h_mask;

	return d_mask;
}