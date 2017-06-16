
/*---------------------------------------------------------------------------*/

// This script is the main function for linear solving;
// Written by Zhipeng Wang, March 8th, 2015

/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "GaussianEL.h"
#include <time.h>

int main(int argc, char **argv){

  //Set the number of thread used:
  
  int num;
  num = atoi(argv[1]);


  // Initialize the row number and column numbers
  int Nrow = 8000;
  int Ncol = 8000;
  // For timing the specific region;
  time_t time_start, time_end;
  double seconds;
   
  // Initialize the vectors, b and x;

  // set up matrix A;
  
  double **A = setup_Am(Nrow, Ncol, num);


  int i,j;
 
 
  for(i=0; i<Nrow; i++){
    for(j=0; j<Ncol; j++){
      
      double rannum = drand48();
      while(rannum < 0.01){
	 	rannum = drand48();
      }      
      A[i][j] = rannum;

    }
    
  }
  

 // set up vector b;

  double *b = setup_bx(Nrow);

  for(i=0; i<Nrow; i++){

    double rannum = drand48();
    b[i] = rannum;

  }


  // set up vector x;
 
  double *x = setup_bx(Nrow);


  // Making a copy of matrix A and b for later checking;

  double **Ao = setup_Am(Nrow, Ncol, num);


 

  for(i=0; i<Nrow; i++){

    memcpy(Ao[i], A[i], Ncol*sizeof(double));

  }

  double *bo = setup_bx(Nrow);

  for(i=0; i<Nrow; i++){

    bo[i] = b[i];

  }
  
  time(&time_start);
   
  // Gaussian Elimination;

  gaussian_el(A, b, Nrow, Ncol, num);

  // Final back substitution

  back_sub(A, b, x, Nrow, Ncol);

  time(&time_end);

  // Checking the correctness of the resulting x;

  double *l2 = setup_bx(Nrow);

  double error = final_checking (Ao, bo, x, Nrow, Ncol, l2);
   

  printf("The error is: %f\n", error);

  seconds = difftime(time_end, time_start);
  printf("The running time is: %f\n", seconds);

  return 0;
}
