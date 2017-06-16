
/* -------------------------------------------------------------------------- */
// This function is trying to parallize the back subsitution process 
// of solving linear equation Ax = b by Pthread

// Input: the A, b in echelon form, x in its initial form;
// Output: x in its final (inverted solution) form;
// Written by Zhipeng Wang, March 8th 2015

/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <math.h>
#include "GaussianEL.h"
#include <pthread.h>


int back_sub(double **A, double *b, double *x, int Nrow, int Ncol){

 
  int i, threadw;

  Thread_args args[NUM_THREADS];


  void *parabackcompt (void *);

  for(i=Nrow-1; i>=0; i--){

    double btemp = b[i];

    // This is the load portion for each thread, we would like to round it up to an integer
    int portion = ceil((double)(Nrow-i) / NUM_THREADS);


    // Count for the number of threads created;
    int thread_count = 0;


    for (threadw=0; threadw<NUM_THREADS; threadw++){

      args[threadw].A = A;
      args[threadw].x = x;
      args[threadw].xtemp = 0;
      args[threadw].index = i;

      // Here we actually check if the lower bound reaches the limit of Nrow, if that is the case, we don't need to create new thread
      if (i + threadw * portion < Nrow){

	// so the lower bound does not reach Nrow, we need to add one to the number of threads created;
	thread_count ++;
      

	args[threadw].jlow = i + threadw * portion;

	if (i + (threadw+1) * portion <= Nrow - 1){
	  args[threadw].jup = i + (threadw+1) * portion;
	}
	else{
	  args[threadw].jup = Nrow - 1;
	}


	// Here we start to create the threads;
	pthread_create(&p_threads[threadw], NULL, parabackcompt, (void*) &args[threadw]);

      }

    }
    
    for (threadw=0; threadw<thread_count; threadw++){

    
      pthread_join(p_threads[threadw], NULL);
     
      btemp -= args[threadw].xtemp;
     

 
    }

    // Divided by the coefficient of x[i];      
    btemp /= A[i][i];
            
    x[i] = btemp;

         
  }
  
  return 0;
}

void *parabackcompt (void *args){


  Thread_args* th_args = (Thread_args *) args;
    
  int j;

 
  for(j=th_args->mupper; j>th_args->mlow; j--){

    th_args->xtemp += th_args->A[th_args->index][j] * th_args->x[j];

  }

  pthread_exit(0);
}
