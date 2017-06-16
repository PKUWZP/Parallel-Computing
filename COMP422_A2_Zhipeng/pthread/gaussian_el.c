
/* -------------------------------------------------------------------------- */

// This function does the Gaussian Elimination of the Ax=b linear 
// equation.
// Input: The matrix A and vector b;
// Output: The matrix A and vector b in echelon form;

// Written by Zhipeng Wang, March 8th, 2015

/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "GaussianEL.h"

int gaussian_el(double **A, double *b, int Nrow, int Ncol, int num){

  int i, j, k;

  int threadw;

  pthread_t p_threads[num];




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

    // Swap the temp line with the ith line;


    double *temp1 = (double*)
      calloc(Ncol,sizeof (double));

    memcpy(temp1,A[i],Ncol*sizeof(double));
    memcpy(A[i],A[temp],Ncol*sizeof(double));
    memcpy(A[temp],temp1,Ncol*sizeof(double));
 
    double temp2 = b[i];
    b[i] = b[temp];
    b[temp] = temp2;

 
    // Do the Gaussian elimination;

    void *elimination (void *);

    gaussian_Thread_args args[num];
 
    // Round up to an interger for the load portion of each thread
    int portion = ceil((double)(Nrow - (i+1)) / num);


     // Count for the number of threads created;
    int thread_count = 0;

    for (threadw=0; threadw<num; threadw++){


   
      args[threadw].index = i;
      args[threadw].Ncol = Ncol;
      args[threadw].A = A;
      args[threadw].b = b;

      args[threadw].Ad = (double*)
	calloc(Ncol, sizeof(double));
      memcpy(args[threadw].Ad, A[i], Ncol*sizeof(double));
      
      args[threadw].bd = b[i];


      if (i+1 + threadw*portion < Nrow){

	// Add one to the number of threads created if it does not cover the full row;
	thread_count ++;
      

	args[threadw].mlow = i+1 + threadw*portion;


	if (i+1 + (threadw+1)*portion < Nrow){
	  args[threadw].mhigh = i+1 + (threadw+1)*portion;
	}
	else{
	  args[threadw].mhigh = Nrow;
	}

	pthread_create(&p_threads[threadw],NULL,elimination,(void*)&args[threadw]);

      }
    
    }

    for (threadw=0; threadw<thread_count; threadw++){
  
      pthread_join(p_threads[threadw],NULL);

    }


  }
  
  return 0;
}

void *elimination (void *args){

  gaussian_Thread_args* th_args = (gaussian_Thread_args *) args;

  int i,j;
  double m;


   // Do the elimination for b;

  for (i=th_args->mlow; i<th_args->mhigh; i++){

    m = th_args->A[i][th_args->index] / th_args->Ad[th_args->index];

    th_args->b[i] -= m * th_args->bd;

    // Do the elimination for A;
    for (j=th_args->index; j<th_args->Ncol; j++){

      th_args->A[i][j] -= m * th_args->Ad[j];

    }

  }
  
  pthread_exit(0);
}
