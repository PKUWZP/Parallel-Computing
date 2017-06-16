/*---------------------------------------------------------------------------*/
// Header file for linear solving problem by using Gaussian 
// elimination. It defines two functions for Gaussian Elimination and back substitution

// Written by Zhipeng Wang, March 8th, 2015
/* --------------------------------------------------------------------------*/

#ifndef SYSTEM_H
#define SYSTEM_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Setting up the matrix A and vector b needed for the linear solving;

double **setup_Am(int Nrow, int Ncol, int num){


  // Allocating the space for the pointers designating the row number of the
  // given matrix;

  double **M = (double**)
    calloc(Nrow, sizeof(double*));

  int i;

  // Making sure the matrix is not stored on just one core;

#pragma omp parallel for default(none) shared(Nrow,M,Ncol) private(i) num_threads(num)
 
    for (i=0; i<Nrow; i++){
      
      M[i] = (double*)
	calloc(Ncol, sizeof(double));
   
    }

  return M;
}



double *setup_bx(int Nrow){

  double *b = (double*)
    calloc(Nrow, sizeof(double));

  return b;

}


// Gaussian-elimination;

int gaussian_el(double **A, double *b, int Nrow, int Ncol, int num);


// Back Substitution

int back_sub(double **A, double *b, double *x, int Nrow, int Ncol);

// Final checking of the results

double final_checking(double **Ao, double *bo, double *x, int Nrow, int Ncol, double *l2);


#endif //SYSTEM_H
