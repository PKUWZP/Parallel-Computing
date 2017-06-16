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



////////////////////////////////////////////////////////////////////////////////
// Shortcut definition
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;



///////////////////////////////////////////////////////////////////////////////
// Sort result validation routines
////////////////////////////////////////////////////////////////////////////////
//Sorted keys array validation (check for integrity and proper order)
extern "C" uint validateSortedKeys(
    uint *resKey,
    uint *srcKey,
    uint batchSize,
    uint arrayLength,
    uint numValues,
    uint dir
);


////////////////////////////////////////////////////////////////////////////////
// CUDA sorting networks
////////////////////////////////////////////////////////////////////////////////

extern "C" uint bitonicSort(
    uint *d_DstKey,
    uint *d_SrcKey,
    uint batchSize,
    uint arrayLength,
    uint dir
);

extern "C" uint bitonicSort2(uint *d_DstKey, uint *d_SrcKey, uint batchSize, uint arrayLength,uint dir);
