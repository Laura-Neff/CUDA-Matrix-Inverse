
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>
struct timeval start_time, stop_time;
int    elapsed;


/* --------------------------------------------------------------
   Help function for your CUDA kernel to print the
   input matrices.


   Make sure that ONLY 1 thread calls this printMatrixPair( )
   function ! (or else, you will see many many outputs !!!)

   In other words, use:

     if ( myUniqueID == 0 )
        printMatrixPair(A,C,n);

   *** If you DID use this help function to debug, you must REMOVE 
   *** the added code when you turn in the program 
   *** (I will test with large matrices and it will cause a lot 
   *** of printing !!!
   ------------------------------------------------------------- */
__device__
void printMatrixPair( float *A, float *C, int n )
{
   for (int i = 0; i < n; i++ )
   {
      for (int j = 0; j < n; j++ )
      {
         printf("%6.2f ", A[i*n+j]);
      }
      printf("  |  ");
      for (int j = 0; j < n; j++ )
      {
         printf("%6.2f ", C[i*n+j]);
      }

      printf("\n");
   }
}


/* =======================================================================
   TODO: write a CUDA kernel to compute the inverse of matrix A

   inverse(A, C, N):

      input:  A = an NxN matrix that you need to find the inverse
                  When the inverse( ) function completes, A will
                  contains the identity matrix
              C = initially contains the identity matrix
                  When the inverse( ) function completes, C will
                  contains the inverse of A
              N = #row (and # columns) in A and C
   ======================================================================== */
__global__
void inverse( float *A, float *C, int N)
{
   /* =============================================================
      Hint:
         0. Main() has spawn N thread

      inverse( ) must:

         1. find the thread ID of the CUDA thread that runs this kernel
            Assume this is thread T

         2. Thread T must process every row in the matrices A and C as
            follows:

              (Processing of row R):
              Step 1: Normalize row R

              The work in step 1 is divided as follows:

                  Thread T computes: A[R][T] = A[R][T]/A[R][R]
                                     C[R][T] = C[R][T]/A[R][R]

              Step 2. Use row R to create a column of 0 values
                      in the column R

              The work in step 2 is divided as follows:

                  Thread T adds  -A[T][R]*row(R) to row(T)
      ============================================================= */


}


void printMatrix( float *A, int n )
{
   for (int i = 0; i < n; i++ )
   {
      for (int j = 0; j < n; j++ )
      {
         printf("%6.2f ", A[i*n+j]);
      }
      printf("\n");
   }
}



int main(int argc, char *argv[])
{
  if ( argc <= 1 )
  {
     printf("Usage: %s N (NxN matrix inversion)\n\n", argv[0]);
     exit(1);
  }

  int N = atoi( argv[1] );

  float *A, *C, *A_org;

  /* ====================================
     Allocate arrays
     ==================================== */
  cudaMallocManaged(&A, N*N*sizeof(float));
  A_org = (float*) calloc(N*N, sizeof(float));
  cudaMallocManaged(&C, N*N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
     for (int j = 0; j < N; j++)
     {
        A[i*N+j] = rand()/1000000000.0;
        A_org[i*N+j] = A[i*N+j];

        if ( i == j )
           C[i*N+j] = 1.0;
        else
           C[i*N+j] = 0.0;
     }

  if ( N <= 5 )
  {
     printf("Input matrix:\n");
     printMatrix( A, N );
     printf("\n");
  }

  gettimeofday(&start_time, NULL);   // Record current sys time as start_time
  /* ========================================================
     CPU algorithm for matrix inversion using Kramer's rule
     ======================================================== */
  // ==================================================================
  // Run kernel on the GPU using 1 block, N thread/per block
  inverse<<<1, N>>>( A, C, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // ==================================================================


   gettimeofday(&stop_time, NULL);   // Record current sys time as stop_time

   elapsed = (stop_time.tv_sec*1000000 + stop_time.tv_usec) -
                (start_time.tv_sec*1000000 + start_time.tv_usec);
   printf("Elasped time = %d micro secs\n", elapsed);

   if ( N <= 5 )
   {
      printf("Matrix A:\n");
      printMatrix( A_org, N );
      printf("\n\nInverse Matrix:\n");
      printMatrix( C, N );
   }
   else
   {
      printf("N > 5; skip printing...\n");
   }

   /* ====================================================
      Check if inverse is correct
      ==================================================== */
  int numErr = 0;

  for (int i = 0; i < N; i++)
     for (int j = 0; j < N; j++)
     {
        // Compute i,j-th element of A * Ainv

        float sum = 0;

        for ( int k = 0; k < N; k++ )
           sum += A_org[i*N+k] * C[k*N+j];

//      printf("Mult[%d][%d] = %f\n", i, j, sum);

        if ( i == j )
        {
           if ( fabs(sum - 1.0) > 0.15 )  numErr++;
        }
        else
        {
           if ( fabs(sum - 0.0) > 0.15 )  numErr++;
        }
     }

   if ( numErr > 0 )
      printf("**** Inverse is NOT correct !\n");
   else
      printf("Inverse is correct !\n");

   // Free memory
   cudaFree(A);
   cudaFree(C);
   free(A_org);

  return 0;
}

