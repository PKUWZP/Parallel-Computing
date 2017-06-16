/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This sample implemenets bitonic sort, algorithms
 * belonging to the class of sorting networks.
 * While generally subefficient on large sequences
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 *
 * Victor Podlozhnyuk, 07/09/2009
 */

/* ---------------------------------------------------------------------------------------------------------------- 
This modified version can sort a specific length of array that is larger than the holding capacity of the GPU (here we assume the capacity is 1M, though the real capacity may be much larger, like 6GB).
   Written by Zhipeng Wang, May 9th, 2015;
------------------------------------------------------------------------------------------------------------------ */

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>
#include <stdlib.h>

#include "sortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  cudaError_t error;
  printf("%s Starting...\n\n", argv[0]);

  printf("Starting up CUDA context...\n");
  int dev = findCudaDevice(argc, (const char **)argv);

  // Get 'kexp' as the input parameter;
  uint kexp = atoi(argv[1]);

  uint *h_InputKey, *h_OutputKeyGPU;
  uint *d_InputKey, *d_OutputKey;
  uint *t_InputKey, *t_OutputKey;
  StopWatchInterface *hTimer = NULL;

  const uint             N = pow(2,kexp);
  const uint             Nload = pow(2,20); // This is for 1M of loading data;
  uint           DIR = 0;
  const uint     numValues = 65536;
    
  // Set up shared-size-limit;

  printf("Allocating and initializing host arrays...\n\n");
  sdkCreateTimer(&hTimer);
  h_InputKey     = (uint *)malloc(N * sizeof(uint));
  h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
  t_InputKey = (uint *)malloc(Nload/2 * sizeof(uint));
  t_OutputKey = (uint *)malloc(Nload/2 * sizeof(uint));

  srand(2001);

  for (uint i = 0; i < N; i++)
    {
      h_InputKey[i] = rand() % numValues;
    }


  // Flag for validating the final sorting result; 
  int flag = 1;
   
  if (kexp<=20){
    // If the input data is smaller than the capacity of the GPU, we just use the existing script for sorting;

    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void **)&d_InputKey,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
    checkCudaErrors(error);

 
    // Change the arraylength to be the same as the given number of unsorted data, we stop doing the looping here and only sort on a specified size of vector;
    uint arrayLength = N;
      
    printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    uint threadCount = 0;

    threadCount = bitonicSort(
			      d_OutputKey,
			      d_InputKey,
			      N / arrayLength,
			      arrayLength,
			      DIR
			      );
    
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);

    sdkStopTimer(&hTimer);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer));

    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
	   (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);
 

    printf("\nValidating the results...\n");
    printf("...reading back GPU results\n");
    error = cudaMemcpy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
    checkCudaErrors(error);

    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
    flag = flag && keysFlag;

    printf("\n");
  }

  else{

    printf("Allocating and initializing CUDA arrays...\n\n");
    // For this part, the loading data into the GPU cannot exceed 1M, so we choose the largest one;
    error = cudaMalloc((void **)&d_InputKey,  Nload * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, Nload * sizeof(uint));
    checkCudaErrors(error);

    uint arrayLength = Nload;
    uint threadCount = 0;

    printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, Nload / arrayLength);

    // Firstly, we need to sort each subsection of the overall array, each subsection having 1M space for data;
    for(int i=0; i<N/Nload; i++){ 

      // Copy the relevant section of the hinputarray into the sorting d array;
      t_InputKey = h_InputKey + Nload*i;
      t_OutputKey = h_OutputKeyGPU + Nload*i;
      error = cudaMemcpy(d_InputKey, t_InputKey, Nload * sizeof(uint), cudaMemcpyHostToDevice);
      checkCudaErrors(error);


      // Change the arraylength to be the same as the given number of unsorted data, we stop doing the looping here and only sort on a specified size of vector;
      uint arrayLength = Nload;
 
      error = cudaDeviceSynchronize();
      checkCudaErrors(error);

      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);
      uint threadCount = 0;

      threadCount = bitonicSort(
				d_OutputKey,
				d_InputKey,
				Nload / arrayLength,
				arrayLength,
				DIR
				);

      error = cudaDeviceSynchronize();
      checkCudaErrors(error);

      // copy the d_array back into the h_array;
      error = cudaMemcpy(t_OutputKey, d_OutputKey, Nload * sizeof(uint), cudaMemcpyDeviceToHost);
      checkCudaErrors(error);

      // Need to change the direction of each chunk of the array consecutively to ensure bitonicity;
      DIR = !(DIR);

    }


    // Copy the h output array back into the h input array in order to prepare for the next stage of operation;
    memcpy(h_InputKey, h_OutputKeyGPU, N * sizeof(uint));

    // Next, we need to sort the array in a larger scale, what we do is to sort chunk by chunk;
    // The loop is for the bitonic sorting network, each chunk has increasing size;
    for (int i=1; i<=kexp-20; i++){
      // The loop is for the bitonic merge chunk, each chunk has decreasing size;
      // Note we need to put p=0 into the loop in order to make sure the last bitonic merging operation is for the case where the loading data in the d_inputkey array is a continuous region in the h_inputkey array memory; in this way the last calling of bitonicsort function ensure the array is completely sorted at that moment;
      for (int p=i; p>=0; p--){

	int size = Nload * pow(2,p);
	int stride = size/2;
	DIR = 0;

	// The loop is for the number of merging chunk we need to sort; note each chunk has different sorting order with respect to its neighbors (to ensure bitonicity); However, the last one should have the same order with respect to its neighbors;
	for (int q=0; q<N/size; q++){

	  // This loop is for the number of loading within each merging chunk, since we have to load 1M by 1M into the bitonicsort function in this sorting process;
	  for (int j=0; j<(stride/(Nload/2)); j++){

	    // The first half chunk;
	    t_InputKey = h_InputKey + q*size + j*Nload/2;


	    error = cudaMemcpy(d_InputKey, t_InputKey, Nload/2*sizeof(uint), cudaMemcpyHostToDevice);
	    checkCudaErrors(error);
	    // The second half of the chunk;
	    t_InputKey = h_InputKey + q*size + j*Nload/2 + stride;


	    d_InputKey += Nload/2;
	    error = cudaMemcpy(d_InputKey, t_InputKey, Nload/2*sizeof(uint), cudaMemcpyHostToDevice);
	    checkCudaErrors(error);
	    // recovering the d_inputKey to its origin;
	    d_InputKey -= Nload/2;


	    
	    if (p>0){
	      // If p is bigger than 0, meaning the stride >= 1M, we only need to sort the array once since the other steps are already set up;
	      threadCount = bitonicSort2(
					 d_OutputKey,
					 d_InputKey,
					 Nload / arrayLength,
					 arrayLength,
					 DIR
					 );
	    }
	    else{
	      // If p is equal to 0, we need to use original bitonicSort function to sort the array thoroughly;
	      threadCount = bitonicSort(
					d_OutputKey,
					d_InputKey,
					Nload / arrayLength,
					arrayLength,
					DIR
					);
	    }

	    // copy the d_array back into the h_array;
	    t_OutputKey = h_OutputKeyGPU + q*size + j*Nload/2;
	    error = cudaMemcpy(t_OutputKey, d_OutputKey, Nload/2*sizeof(uint), cudaMemcpyDeviceToHost);
	    checkCudaErrors(error);


	    t_OutputKey = h_OutputKeyGPU + q*size + j*Nload/2 + stride;
	    d_OutputKey += Nload/2;
	    error = cudaMemcpy(t_OutputKey, d_OutputKey, Nload/2*sizeof(uint), cudaMemcpyDeviceToHost);
	    checkCudaErrors(error);
	    d_OutputKey -= Nload/2;

	  }


	  // Need to change the direction of each chunk of the array consecutively to ensure bitonicity (except for the last one, the largest outerloop);
	  if (i != kexp-20){
	    DIR = !(DIR);
	  }


	}
	
	// Before processing the next level of bitonic merging, we need to copy the H output array into the H input array to update the H array before starting another run;
	memcpy(h_InputKey, h_OutputKeyGPU, N * sizeof(uint));

      }
    }


    sdkStopTimer(&hTimer);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer));

    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
	   (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);

    printf("\nValidating the results...\n");

    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
    flag = flag && keysFlag;

    printf("\n");


  }

  printf("Shutting down...\n");
  
  sdkDeleteTimer(&hTimer);
  cudaFree(d_OutputKey);
  cudaFree(d_InputKey);
  free(h_OutputKeyGPU);
  free(h_InputKey);

  // free(t_InputKey); 
  // free(t_OutputKey);

  cudaDeviceReset();
  exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
