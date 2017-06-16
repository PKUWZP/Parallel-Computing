
/*--------------------------------------------------------------------------*/

// This function will check for the correctness of the results 

// written by Zhipeng Wang, March 8th, 2015

/*--------------------------------------------------------------------------*/

#include <stdio.h>
#include <math.h>

double final_checking(double **Ao, double *bo, double *x, int Nrow, int Ncol, double *l2){

  int i, j;


  for (i=0; i<Nrow; i++){

    double temp = 0;

    for (j=0; j<Ncol; j++){

      temp += Ao[i][j]*x[j];

    }

    l2[i] = temp - bo[i];

  }
  

  double err = 0;

  for(i=0; i<Nrow; i++){

    err += pow(l2[i],2);

  }

  err = sqrt(err);

  return err;

}
