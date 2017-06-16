
/* -------------------------------------------------------------------------- */

// This function will do the Gaussian Elimination of the Ax=b linear 
// equation.
// Input: The matrix A and vector b;
// Output: The matrix A and vector b in echelon form;

// Written by Zhipeng Wang, March 8th, 2015


/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

int gaussian_el(double **A, double *b, int Nrow, int Ncol, int num){

  int i, j, k;



  for (i=0; i<Nrow; i++){


    int temp = i;
    double temp_abs = fabs(A[i][i]);
 
    for (j=i+1; j<Nrow; j++){


 
      // compare and find the line with the minimum abosolute column j value;

      if (fabs(A[j][i]) < temp_abs){
	

	temp_abs = A[j][i];
	temp = j;

      }

    }
  

    // Swapping the temp line with the ith line;
    double *temp1 = (double*)
      calloc(Ncol,sizeof (double));

    memcpy(temp1,A[i],Ncol*sizeof(double));
    memcpy(A[i],A[temp],Ncol*sizeof(double));
    memcpy(A[temp],temp1,Ncol*sizeof(double));
 
    double temp2 = b[i];
    b[i] = b[temp];
    b[temp] = temp2;



    // Do the Gaussian elimination;

#pragma omp parallel for shared(A,b,i,Ncol,Nrow) private(k,j) default(none) num_threads(num) 

    for (k=i+1; k<Nrow; k++){
      
      double m = A[k][i]/A[i][i];

      b[k] -= m * b[i];

      for(j=i; j<Ncol; j++){

	A[k][j] -= m * A[i][j];

      }
     
    }

   
  }

  return 0;

}
