
/* -------------------------------------------------------------------------- */
// This function is doing back substitution and does the final steps
// of solving linear equation Ax = b;

// Input: the A, b in echelon form, x in its initial form;
// Output: x in its final (inverted solution) form;
// Written By Zhipeng Wang, March 8th, 2015

/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <math.h>


int back_sub(double **A, double *b, double *x, int Nrow, int Ncol){

  int i, j;


  double temp_x;

  for(i=Nrow-1; i>=0; i--){


    temp_x = b[i]; //temporal variable of x

    //    #pragma omp parallel for shared(A,x,Nrow,i) private(j) reduction(-: xtemp) default(none) num_threads(32) 
    
    for(j=Nrow-1; j>i; j--){
      
      temp_x -= A[i][j]*x[j];
      
    }	 

    // Divided by the coefficient of x[i];
      temp_x /= A[i][i];

      
      x[i] = temp_x;
      
    
  }
  
  return 0;

}
