/* -------------------------------------------------------------------------- */
// This is the function for doing the matrixproduction;
// Written by Zhipeng Wang,  May 4th, 2015;
/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>

void matrixproduct(int n, double *a, double *b, double *tC){

  int i,  j, k;

  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      for(k=0; k<n; k++){
	tC[i*n+j] += a[i*n+k] * b[k*n+j];
      }
    }
  }

} 
