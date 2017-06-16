/*---------------------------------------------------------------------------*/
// Header file introduces two functions, for Gaussian elimination and 
// final back substitution. It also gives the definition of matrix and vector used
// in this file.
// Written by Zhipeng Wang, March 8th, 2015
/* --------------------------------------------------------------------------*/

#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdio.h>
#include <stdlib.h>

// Setting up the A and b needed for the linear solving;

double **setup_Am(int Nrow, int Ncol);


double *setup_bx(int Nrow);


void free_matrix(double **M);

void free_array(double *b);


// Gaussian-elimination;

int gaussian_el(double **A, double *b, int Nrow, int Ncol, int num);


// Back substitution;

int back_sub(double **A, double *b, double *x, int Nrow, int Ncol);

// final checking the result;

double final_checking(double **Ao, double *bo, double *x, int Nrow, int Ncol, double *l2);

// struct for the arguments in each thread function;
typedef struct {

  int mlow;
  int mupper;
  double xtemp;
  double **A;
  double *x;
  int index;

} Thread_args;


typedef struct {

  int mhigh;
  int mlow;
  int Ncol;
  double **A;
  double *b;
  int index;
  double* Ad; // For storing the A's specific row;
  double bd; // For storing the specific row entry of b array;

} gaussian_Thread_args;





#endif //SYSTEM_H
