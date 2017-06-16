//This script would build up the matrix A and b for linear solving
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

//setting up pthread parallization
double **setup_Am(int Nrow, int Ncol){



  double **M = (double**)
    calloc(Nrow, sizeof(double*));

  // setting up space for the entries of the given matrix;

  M[0] = (double*)
    calloc(Nrow*Ncol, sizeof(double));

  // setting up the pointers for row index;

  int i;
  for(i=1;i<Nrow;i++){
    
    M[i] = M[i-1] + Ncol;

  }
  return M;
}

double *setup_bx(int Nrow){

  double *b = (double*)
    calloc(Nrow, sizeof(double));

  return b;

}


void free_matrix(double **M){

  free(M[0]);

  free(M);

}

void free_array(double *b){
  
  free(b);

}
