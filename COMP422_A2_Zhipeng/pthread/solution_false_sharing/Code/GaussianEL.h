/*---------------------------------------------------------------------------*/
// Header file for the linear solving problem by using Gaussian 
// elimination. It builds up two functions, for Gaussian elimination and 
// final solution. It also gives the definition of matrix and vector used
// in this script.

// Written by Zhipeng Wang, March 8th, 2015
/* --------------------------------------------------------------------------*/

#ifndef SYSTEM_H
#define SYSTEM_H

#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 2

// Setting up the A and b needed for the linear solving;

double **setup_Am(int Nrow, int Ncol);


double *setup_bx(int Nrow);


void free_matrix(double **M);

void free_array(double *b);


// Gaussian-elimination;

int gaussian_el(double **A, double *b, int Nrow, int Ncol);


// Final solution;

int back_sub(double **A, double *b, double *x, int Nrow, int Ncol);

// checking the result;

double final_checking(double **Ao, double *bo, double *x, int Nrow, int Ncol, double *l2);

// struct for the arguments in each thread function;
typedef struct {

  int jlow;
  int jup;
  double xtemp;
  double **A;
  double *x;
  int index;

} Thread_args;


pthread_t p_threads[NUM_THREADS];


typedef struct {

  int Ncol;
  double **AT;
  double *bt;
  int index;
  int T_num;
  double* Ad; // For storing the A's specific-divided row;
  double bd; // For storing the specific-divided row entry of b array;

} gaussian_Thread_args;





#endif //SYSTEM_H
