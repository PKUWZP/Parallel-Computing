
/* -------------------------------------------------------------------------- */

// This script will help do the Gaussian Elimination of the Ax=b linear 
// equation.
// Input: The matrix A and vector b;
// Output: The matrix A and vector b in echelon form;

// Codes written by Zhipeng Wang, March 8th, 2015 


/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "GaussianEL.h"

int gaussian_el(double **A, double *b, int Nrow, int Ncol){

  int i, j, k;

  int threadw;

  pthread_t p_threads[NUM_THREADS];

  



  for (i=0; i<Nrow; i++){


    int temp = i;
    double temp_abs = fabs(A[i][i]);
 
    for (j=i+1; j<Nrow; j++){


 
      // compare and find the line with the minimum abosolute value of column j;

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

    // Start to do the gaussian eliminations ;

    void *elimination (void *);

    int thread_count = 0;

    int mlow[NUM_THREADS];
    int mhigh[NUM_THREADS];    

    gaussian_Thread_args args[NUM_THREADS];


    for (threadw=0; threadw<NUM_THREADS; threadw++){

      
      int portion = ceil((double)(Nrow - (i+1)) / NUM_THREADS);

      args[threadw].index = i;
      args[threadw].Ncol = Ncol;

      args[threadw].Ad = (double*)
	calloc(Ncol, sizeof(double));// 
      memcpy(args[threadw].Ad, A[i], Ncol*sizeof(double));



      args[threadw].bd = b[i];

      if((i+1) + threadw*portion < Nrow){

	thread_count ++;

	mlow[threadw] = i+1 + threadw*portion;

	if((i+1) + (threadw+1) * portion < Nrow){

	  mhigh[threadw] = i+1 + (threadw+1) * portion;

	}
	else{

	  mhigh[threadw] = Nrow;

	}



	int T_num = mhigh[threadw] - mlow[threadw];

	args[threadw].T_num = T_num;

	double **AT = (double**)
	  calloc(T_num, sizeof(double*));

	AT[0] = (double*)
	  calloc(T_num*Ncol,sizeof(double));

	int wat;
	for(wat=1; wat<T_num; wat++){

	  AT[wat] = AT[wat-1] + Ncol; 

	}

	for(wat=0; wat<T_num; wat++){

	  memcpy(AT[wat], A[mlow[threadw]+wat], Ncol*sizeof(double));

	}
    


	double *bt = (double*)
	  calloc(T_num, sizeof(double));

	memcpy(bt, &b[mlow[threadw]], T_num*sizeof(double));

	args[threadw].AT = AT;
	args[threadw].bt = bt;


	pthread_create(&p_threads[threadw], NULL, elimination, (void*) &args[threadw]);

      }
    }

      for (threadw=0; threadw<thread_count; threadw++){

	pthread_join(p_threads[threadw],NULL);
	memcpy(A[mlow[threadw]], args[threadw].AT[0], args[threadw].T_num*Ncol*sizeof(double));
	memcpy(&b[mlow[threadw]], args[threadw].bt, args[threadw].T_num*sizeof(double));
	free_matrix(args[threadw].AT);
	free_array(args[threadw].bt);

      }

  }
    
  return 0;
}

void *elimination (void *args){

  gaussian_Thread_args* th_args = (gaussian_Thread_args *) args;

  int k,j;
  double m;



   // Do the gaussian elimination for b;

  for (k=0; k<th_args->T_num; k++){

    m = th_args->AT[k][th_args->index] / th_args->Ad[th_args->index];


    th_args->bt[k] -= m * th_args->bd;

    // Do the gaussian elimination for A;
    for (j=th_args->index; j<th_args->Ncol; j++){

      th_args->AT[k][j] -= m * th_args->Ad[j];

    }

  }
  
  pthread_exit(0);
}
