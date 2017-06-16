
/* -------------------------------------------------------------------------- */
// This function does the back substitution and accomplishes the final part
// of solving linear equation Ax = b;

// Input: the A, b in echelon form, x in its initial form;
// Output: x in its final (inverted solution) form;

// Codes written by Zhipeng Wang, March 8th, 2015

// I do not do the parallization here

/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <math.h>
#include "GaussianEL.h"
#include <pthread.h>


int back_sub(double **A, double *b, double *x, int Nrow, int Ncol){


  int i, j;

  for(i=Nrow-1; i>=0; i--){


    x[i] = b[i];
   
    for(j=Nrow-1; j>i; j--){
  
      x[i] -= A[i][j]*x[j];
      
    }	 

    // Divided by the coefficient of x[i];
    x[i] /= A[i][i];

  }
  return 0;

}
