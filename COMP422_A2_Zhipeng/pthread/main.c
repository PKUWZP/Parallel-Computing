
/*---------------------------------------------------------------------------*/

// This is the main function for the linear solving;
// Written by Zhipeng Wang, March 8th, 2015

/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "GaussianEL.h"
#include <time.h>

int main(int argc, char **argv){
 
  //Set the number of thread used:
  
  int num;
  num = atoi(argv[1]);

  // Set the row number and column number;

  int Nrow = 8000;
  int Ncol = 8000;
  // For timing the specific region;
  time_t time_start, time_end;
  time_t time_inter;
  double seconds;

  
  // Initialize the matrix, b and x;

  // set up matrix A;
  
  double **A = setup_Am(Nrow, Ncol);

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

  double **Ao = setup_Am(Nrow, Ncol);

  for(i=0; i<Nrow; i++){

    for(j=0; j<Ncol;j++){

      Ao[i][j] = A[i][j];

    }

  }

  double *bo = setup_bx(Nrow);

  for(i=0; i<Nrow; i++){

    bo[i] = b[i];

  }
  
  time(&time_start);   
  // Gaussian Elimination;

  gaussian_el(A, b, Nrow, Ncol, num);

  time(&time_inter);
  seconds = difftime(time_inter, time_start);
  printf("The intermediate time is: %f\n", seconds);

  // Final solution by back substitution;

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
