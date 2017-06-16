/* -------------------------------------------------------------------------- */

// This code would do the matrix production using 2.5D algorithm;
// Finished by Zhipeng Wang, May 7th, 2015 for COMP 422

/* -------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "header.h"

#define max(i,j) (i > j ? i : j)
#define min(i,j) (i < j ? i : j)



int main(int argc, char **argv){


  double time_begin, time_end;
  MPI_Init(&argc, &argv);

  int X = 0, Y = 1, Z = 2;
  int i, j, k;
  int p, q;
  int nlocal;

  // Checking flag to see if MPI function works well;
  int checkpoint;

  // 3D parameters;
  int npes, dims[3], periods[3], keep_dims[3];
  int myrank, my3drank, mycoords[3];
  int other_rank, coords[3];
  MPI_Status status;
  MPI_Comm comm_3d, comm_X, comm_Y, comm_Z;

  // 2D Cannon parameters;
  double *a_buffers[2], *b_buffers[2];
  int my2drank;
  int uprank, downrank, leftrank, rightrank;
  int coord_diff;
  int shiftsource, shiftdest;
  MPI_Comm comm_2d;
  MPI_Request reqs[4];

  // Set up matrix information;

  double **A;
  double **B;
  double **C;
  double **tC;
  double **result;
  
  // specify the dimension of the matrix and the length of the extended dimension in the new topology;
  int Nrow;

  int length;

  // Get information about the communicator;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  if (myrank==0){

    Nrow = atoi(argv[1]);
    length = atoi(argv[2]);

  }
  
  MPI_Bcast(&Nrow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  // Set up Cartesian topology and get the rank & coordinates of the process in this topology;

  dims[X] = dims[Y] = sqrt(npes/length);
  dims[Z] = length;


  nlocal = Nrow / sqrt(npes/length);

  periods[X] = periods[Y] = periods[Z] = 1; // Set the periods for wrap-around connections;
  
  // Get the new communicator in the new topolgy
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &comm_3d);

  // Get the rank in the new topology;
  MPI_Comm_rank(comm_3d, &my3drank);
  
  // Get the coordinates in the new topology;
  
  MPI_Cart_coords(comm_3d, my3drank, 3, mycoords);

  // Setting up matrix A, B and C; Every processor should setup A and B, while only the processors at the front face of the grid need to initialize them and do the broadcast to other processors;
    
  A = (double**)
    calloc(nlocal, sizeof(double*));

  A[0] = (double*)
    calloc(nlocal*nlocal, sizeof(double));

  for(q=1; q<nlocal; q++){

    A[q] = A[q-1] + nlocal;

  }
  
  B = (double**)
    calloc(nlocal, sizeof(double*));

  B[0] = (double*)
    calloc(nlocal*nlocal, sizeof(double));

  for(q=1; q<nlocal; q++){

    B[q] = B[q-1] + nlocal;

  }
  
  C = (double**)
    calloc(nlocal, sizeof(double*));

  C[0] = (double*)
    calloc(nlocal*nlocal, sizeof(double));

  for(q=1; q<nlocal; q++){

    C[q] = C[q-1] + nlocal;

  }

  tC = (double**)
    calloc(nlocal, sizeof(double*));

  tC[0] = (double*)
    calloc(nlocal*nlocal, sizeof(double));

  for(q=1; q<nlocal; q++){

    tC[q] = tC[q-1] + nlocal;

  }
  
  result = (double**)
    calloc(nlocal, sizeof(double*));

  result[0] = (double*)
    calloc(nlocal*nlocal, sizeof(double));

  for(q=1; q<nlocal; q++){

    result[q] = result[q-1] + nlocal;

  }
 
  
  // Set up the buffer arrays for non-blocking communication;
  a_buffers[0] = A[0];
  a_buffers[1] = (double*)
    calloc(nlocal*nlocal, sizeof(double));
  b_buffers[0] = B[0];
  b_buffers[1] = (double*)
    calloc(nlocal*nlocal, sizeof(double));



  // Create a length based cannon sub-topology;
  /*
  keep_dims[X] = 0;
  keep_dims[Y] = 0;
  keep_dims[Z] = 1;


  MPI_Cart_sub(comm_3d, keep_dims, &comm_Z);
  */
  int color = mycoords[X]*dims[Y] + mycoords[Y];
  MPI_Comm_split(comm_3d, color, mycoords[Z], &comm_Z);

  int myzrank;
  MPI_Comm_rank(comm_Z, &myzrank);

  
  if (mycoords[Z] == 0){

    // Select the first processor of the Z directional communicator to broadcast;
     
    // This means it is the processor in the Z=0 face (initial A and B place);

    //TEST starts here

    for (p=0; p<nlocal; p++){

      for (q=0; q<nlocal; q++){
	
	int tempi = nlocal*mycoords[X] + p;
	int tempj = nlocal*mycoords[Y] + q;

	 
	// TEST 1;

       	A[p][q] = 1 + (tempi + tempj)%Nrow;
	B[p][q] = 1;
	
	result[p][q] = Nrow*(Nrow+1)/2;

      }

    }


  }

  // Calculate the MPI implementation starting time (with the root processor):
  if (myrank == 0){
    // This means it is in the root processor;

    time_begin = MPI_Wtime();
  }
  // Doing the broadcast of the A and B across the X direction according to certain laws;


  // Select the first processor of the Z directional communicator to broadcast;
 
  
  checkpoint = MPI_Bcast(A[0], nlocal*nlocal, MPI_DOUBLE, 0, comm_Z);

  if(checkpoint != MPI_SUCCESS){

    printf("The broadcast failed with the rank: %d\n", my3drank);

  }

  MPI_Bcast(B[0], nlocal*nlocal, MPI_DOUBLE, 0, comm_Z);

  // Perform the initial circular shift on A;
  
  // Based on common visualization, we represent three dimensional coordinates by using i, j and k; Note the wrap-around topology will take care about the MOD calculation;
  i = mycoords[X];
  j = mycoords[Y];
  k = mycoords[Z];

  // Note! the "/" here cause a lot of problem with int/int = 0, which is not we want, so we change it to int/(double);
  int rdecider = j + i - k/(double)length*sqrt(npes/length);
  if (rdecider < 0)
    rdecider += dims[X];

  int r = rdecider % (int)sqrt(npes/length);

  coords[X] = i;
  coords[Y] = r;
  coords[Z] = k;

  // Note this is to calculate the shift source and destination

  MPI_Cart_rank(comm_3d, coords, &shiftsource);

  int s1decider = j - i + k/(double)length*sqrt(npes/length);
  if (s1decider < 0)
    s1decider += dims[X];

  int s1 = s1decider % dims[X];

  coords[X] = i;
  coords[Y] = s1;
  coords[Z] = k;

  MPI_Cart_rank(comm_3d, coords, &shiftdest);


  checkpoint = MPI_Sendrecv_replace(a_buffers[0], nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_3d,  &status);

  if(checkpoint != MPI_SUCCESS){

    printf("The sendrecv&replace failed with the rank: %d\n", my3drank);

  }

  
  
  // Perform the initial circular shift on B; Similarly, the wrap-around topology takes care of the mod calculation;

  coords[X] = r;
  coords[Y] = j;
  coords[Z] = k;

  MPI_Cart_rank(comm_3d, coords, &shiftsource);

  int s2decider = i - j + k/(double)length*sqrt(npes/length);
  if (s2decider < 0)
    s2decider += dims[X];

  int s2 = s2decider % dims[X];

  coords[X] = s2;
  coords[Y] = j;
  coords[Z] = k;

  MPI_Cart_rank(comm_3d, coords, &shiftdest);

  
  checkpoint = MPI_Sendrecv_replace(b_buffers[0], nlocal*nlocal, MPI_DOUBLE, shiftdest, 1, shiftsource, 1, comm_3d, &status);

  if(checkpoint != MPI_SUCCESS){

    printf("The sendrecv&replace failed with the rank: %d\n", my3drank);

  }
  
  // Do the Cannon analogous shifting and adding operation;
  // For left <-> right shifting, we need to perform the Y direction shifting;;
  // For Up <-> Down shifting, we need to perform the X direction shifting;

  MPI_Cart_shift(comm_3d, Y, -1, &rightrank, &leftrank);
  MPI_Cart_shift(comm_3d, X, -1, &downrank, &uprank);
    
  if (dims[X]/(double)length > 1){

    //Then we need to step into the Cannon shift loop, so we should not ignore the Isend and Irecv part here;
    MPI_Isend(a_buffers[0], nlocal*nlocal, MPI_DOUBLE, leftrank, 1, comm_3d, &reqs[0]);
    MPI_Isend(b_buffers[0], nlocal*nlocal, MPI_DOUBLE, uprank, 1, comm_3d, &reqs[1]);
    MPI_Irecv(a_buffers[1], nlocal*nlocal, MPI_DOUBLE, rightrank, 1, comm_3d, &reqs[2]);
    MPI_Irecv(b_buffers[1], nlocal*nlocal, MPI_DOUBLE, downrank, 1, comm_3d, &reqs[3]);
 
    for (j=0; j<4; j++){
      MPI_Wait(&reqs[j], &status);
    }  
  }
  
  matrixproduct(nlocal, a_buffers[0], b_buffers[0], tC[0]);

  /*
  //  Blocking case:
 
  matrixproduct(nlocal, A[0], B[0], tC[0]);

  */
  //Note the loop starts from one to ensure in sepcial case the 3D matrix product version can be recovered;

  for(i=1; i<dims[X]/length; i++){

    
    MPI_Isend(a_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, leftrank, 1, comm_3d, &reqs[0]);
    MPI_Isend(b_buffers[i%2], nlocal*nlocal, MPI_DOUBLE, uprank, 1, comm_3d, &reqs[1]);
    MPI_Irecv(a_buffers[(i+1)%2], nlocal*nlocal, MPI_DOUBLE, rightrank, 1, comm_3d, &reqs[2]);
    MPI_Irecv(b_buffers[(i+1)%2], nlocal*nlocal, MPI_DOUBLE, downrank, 1, comm_3d, &reqs[3]);

    matrixproduct(nlocal, a_buffers[i%2], b_buffers[i%2], tC[0]);

    for (j=0; j<4; j++){
      MPI_Wait(&reqs[j], &status);
    }
    

    /*
    MPI_Sendrecv_replace(A[0], nlocal*nlocal, MPI_DOUBLE, leftrank, 1, rightrank, 1, comm_3d, &status);
    MPI_Sendrecv_replace(B[0], nlocal*nlocal, MPI_DOUBLE, uprank, 1, downrank, 1, comm_3d, &status);

    matrixproduct(nlocal, A[0], B[0], tC[0]);
    */      
      
  }

  // Perform the sum_reduction along the X directions to get to the final c;

  MPI_Reduce(tC[0], C[0], nlocal*nlocal, MPI_DOUBLE, MPI_SUM, 0, comm_Z);

  // Barrier for the final procedure;
  MPI_Barrier(comm_3d);

  // Calculate the MPI implementation ending time (for the root processor);
  
  if (myrank == 0){
    time_end = MPI_Wtime();

    printf("The time elapsed for this MPI implementation is: %f\n", time_end-time_begin);
  }
  
  // Check and Print the final result;
  int checkflag = 0;

  if (mycoords[Z] == 0){

    // This means it is where the C is stored;
    
    for (p=0; p<nlocal; p++){
      for(q=0; q<nlocal; q++){

	if(C[p][q] != result[p][q]){
	  //	  printf("%f\t%f\n", result[p][q], C[p][q]);
	  //  printf("\n problem with %d, %d\n", p, q);
	  checkflag = 1;

	}
      }
    }

    if (checkflag == 1){

      printf("There is some problem somewhere\n");

    }

  }

  // Free the 3D communicators;

  MPI_Comm_free(&comm_3d);
  MPI_Comm_free(&comm_Z);
  

  MPI_Finalize();

  return 0;
    
}
