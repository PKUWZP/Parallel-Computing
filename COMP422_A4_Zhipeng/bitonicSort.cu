/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm



#include <assert.h>
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"



////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void bitonicSortShared(
				  uint *d_DstKey,
				  uint *d_SrcKey,
				  uint arrayLength,
				  uint dir,
				  uint smallsize
				  )
{
  //Shared memory storage for one or more short vectors
  //It is fine for here to define a vector size larger than necessary if kexp is smaller than 10, because later analysis will find we can only use the partial 'smallsize' of it;
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
 
  //Offset to the beginning of subbatch and load data
  d_SrcKey += blockIdx.x * smallsize + threadIdx.x;
  d_DstKey += blockIdx.x * smallsize + threadIdx.x;
  s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
  s_key[threadIdx.x + (smallsize / 2)] = d_SrcKey[(smallsize / 2)];

  for (uint size = 2; size < arrayLength; size <<= 1)
    {
      //Bitonic merge
      uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

      for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
	  __syncthreads();
	  uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
	  Comparator(
		     s_key[pos +      0],
		     s_key[pos + stride],
		     ddd
		     );
        }
    }

  //ddd == dir for the last bitonic merge step
  {
    for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
      {
	__syncthreads();
	uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
	Comparator( 
		   s_key[pos + 0],
		   s_key[pos + stride], 
		   dir
		    );
      }
  }

  __syncthreads();
  d_DstKey[                      0] = s_key[threadIdx.x +                       0];
  d_DstKey[(smallsize / 2)] = s_key[threadIdx.x + (smallsize / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
//This correspond to arraylength larger than SHARED_SIZE_LIMIT, so there will not be case of small size, so we need not modify here;
__global__ void bitonicSortShared1(uint *d_DstKey, uint *d_SrcKey)
{
  //Shared memory storage for current subarray
  __shared__ uint s_key[SHARED_SIZE_LIMIT];

  //Offset to the beginning of subarray and load data
  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

  // size: size of the bitonic merge network;
  // stride: stride size for the comparison;

  for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
    {
      //Bitonic merge
      uint ddd = (threadIdx.x & (size / 2)) != 0;
 
      for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
	  __syncthreads();
	  uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
	  Comparator(
		     s_key[pos +      0],
		     s_key[pos + stride],
		     ddd
		     );
        }
    }

  //Odd / even arrays of SHARED_SIZE_LIMIT elements
  //sorted in opposite directions
  //This is the bitonic merge for the largest chunk in the bitonic sorting network;
  uint ddd = blockIdx.x & 1;
  {
    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
      {
	__syncthreads();
	uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
	Comparator(
		   s_key[pos +      0],
		   s_key[pos + stride],
		   ddd
		   );
      }
  }


  __syncthreads();
  d_DstKey[                      0] = s_key[threadIdx.x +                       0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
				   uint *d_DstKey,
				   uint *d_SrcKey,
				   uint arrayLength,
				   uint size,
				   uint stride,
				   uint dir
				   )
{
  uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
  uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

  //Bitonic merge
  // ddd switches from 'dir' to 'minus dir' consecutively;
  uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
  uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  uint keyA = d_SrcKey[pos +      0];
  uint keyB = d_SrcKey[pos + stride];

  Comparator(
	     keyA,
	     keyB,
	     ddd
	     );

  d_DstKey[pos +      0] = keyA;
  d_DstKey[pos + stride] = keyB;
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
				   uint *d_DstKey,
				   uint *d_SrcKey,
				   uint arrayLength,
				   uint size,
				   uint dir
				   )
{
  //Shared memory storage for current subarray
  __shared__ uint s_key[SHARED_SIZE_LIMIT];

  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

  //Bitonic merge
  uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);

  uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

  for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
      __syncthreads();
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(
		 s_key[pos +      0],
		 s_key[pos + stride],
		 ddd
		 );
    }

  __syncthreads();
  d_DstKey[                      0] = s_key[threadIdx.x +                       0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
  if (!L)
    {
      *log2L = 0;
      return 0;
    }
  else
    {
      for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

      // If it is not a power of 2, it will not return 1 (may be 0 or some value bigger than 1);
      return L;
    }
}

extern "C" uint bitonicSort(
			    uint *d_DstKey,
			    uint *d_SrcKey,
			    uint batchSize,
			    uint arrayLength,
			    uint dir
			    )
{

  //Nothing to sort
  if (arrayLength < 2)
    return 0;

  //Only power-of-two array lengths are supported by this implementation
  uint log2L;
  uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  uint blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;

  // We need to change here, delete the assertion so that the program can deal with the case of kexp < 10;
  if (arrayLength <= SHARED_SIZE_LIMIT)
    {
      if ((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0){
    
	bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey,  d_SrcKey, arrayLength, dir, SHARED_SIZE_LIMIT);

      }
      
      else{
	// The kexp is smaller than 10;
	uint smallsize = batchSize * arrayLength;
	blockCount = 1;
	threadCount = batchSize * arrayLength / 2;
	bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_SrcKey, arrayLength, dir, smallsize);
	
      }
      
    }
  else
    {
      bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_SrcKey);


      for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
	for (unsigned stride = size / 2; stride > 0; stride >>= 1)
	  if (stride >= SHARED_SIZE_LIMIT)
	    {
	      // if stride>shared_size_limit, we are not able to use just one array to do the bitonic merge, thus we call the bitonic merge function each time the loop come here; and it compare and swap just two values each time, instead of creating an array and sway in it once and for all; 
	      bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstKey, arrayLength, size, stride, dir);

	    }
	  else
	    {
	      bitonicMergeShared<<<blockCount, threadCount>>>(d_DstKey, d_DstKey, arrayLength, size, dir);
	      // Since it is swap once and for all, it need to break after that;
	      break;
	    }
    }


  return threadCount;
}
