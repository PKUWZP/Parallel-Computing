/* ---------------------------------------------------------------------------
   This is an alternative version of Bitonicsort for sorting partial chunk of a vector which is overlarge than the loading limit of the GPU;
   Written by Zhipeng Wang, May 8th, 2015
   ---------------------------------------------------------------------------- */


#include <assert.h>
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"


//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal2(
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


extern "C" uint bitonicSort2(
			    uint *d_DstKey,
			    uint *d_SrcKey,
			    uint batchSize,
			    uint arrayLength,
			    uint dir
			    )
{

  uint factorRadix2(uint *log2L, uint L);

  //Nothing to sort
  if (arrayLength < 2)
    return 0;

  //Only power-of-two array lengths are supported by this implementation
  uint log2L;
  uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);
  
  dir = (dir != 0);

 


  uint size = arrayLength;
  uint stride = size / 2;
  // if stride>shared_size_limit, we are not able to use just one array to do the bitonic merge, thus we call the bitonic merge function each time the loop come here; and it compare and swap just two values each time, instead of creating an array and sway in it once and for all; 
  bitonicMergeGlobal2<<<(batchSize * arrayLength) / 512, 256>>>(d_DstKey, d_DstKey, arrayLength, size, stride, dir);



  uint threadCount = SHARED_SIZE_LIMIT / 2;
  return threadCount;

}

