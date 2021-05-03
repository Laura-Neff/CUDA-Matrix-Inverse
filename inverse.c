
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define  Epsilon 0.15

#include <sys/time.h>
struct timeval start_time, stop_time;
int    elapsed;

void printMatrixPair( float *A, float *C, int n );

void inverse( float *A, float *C, int N)
{
  /* ==========================================================
     R = a variable that goes through each row of the matrix
     ========================================================== */
  for (int R = 0; R < N; R++)
  {
     float factor = A[R*N + R];	// Use A[R][R] as multiply factor

     for (int j = 0; j < N; j++)        // Normalize row R with factor
     {
        A[R*N+j] = A[R*N+j]/factor;
        C[R*N+j] = C[R*N+j]/factor;
     }

     /* -------------------------------------------------
        Print progress if matrix size is small enough
        ------------------------------------------------- */
     if ( N <= 5 )
     {
        printf("After normalizing row %d:\n", R);
        printMatrixPair( A, C, N );
        printf("\n");
     }

     /* =========================================================
        Make a column of 0 values in column R using the row "R"
        ========================================================= */
     for (int i = 0; i < N; i++)
     {
        if (i == R)
        {
           // Do nothing to row "R"
        }
        else
        {
           float f = A[i*N+R];		// Multiply factor

           /* -------------------------------------
              Add  -f*row(R) to row(i) 
              ------------------------------------- */
           for (int j = 0; j < N; j++)
           {
              A[i*N+j] = A[i*N+j] - f*A[R*N+j];
              C[i*N+j] = C[i*N+j] - f*C[R*N+j];
           }

           /* -------------------------------------------------
              Print progress if matrix size is small enough
              ------------------------------------------------- */
           if ( N <= 5 )
           {
              printf("After reducing row %d:\n", i);
              printMatrixPair( A, C, N );
              printf("\n");
           }
        }
     }

     if ( N <= 5 )
        printf("=============================================\n");
   }

}


void printMatrixPair( float *A, float *C, int n )
{
   for (int i = 0; i < n; i++ )
   {
      for (int j = 0; j < n; j++ )
      {
         printf("%7.3f ", A[i*n+j]);
      }
      printf("  |  ");
      for (int j = 0; j < n; j++ )
      {
         printf("%7.3f ", C[i*n+j]);
      }

      printf("\n");
   }
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
     printf("To enter a specific matrix, use: %s N manual\n\n", argv[0]);
     exit(1);
  }

  int N = atoi( argv[1] );
  int K;

  float *A, *C, *A_org;

  /* ====================================
     Allocate arrays
     ==================================== */
  A = calloc(N*N, sizeof(float));
  A_org = calloc(N*N, sizeof(float));
  C = calloc(N*N, sizeof(float));

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

  if ( argc == 3 && strcmp(argv[2],"manual")== 0 )
  {
     printf("Manual input of a %dx%d matrix:\n\n", N, N);

     for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
           printf("A[%d][%d] = ", i, j);
           scanf("%f", &A[i*N+j]);
           A_org[i*N+j] = A[i*N+j];
        }
  }

  if ( N <= 5 )
  {
     printf("Initial state:\n");
     printMatrixPair( A, C, N );
     printf("\n");
  }

  /* ========================================================
     CPU algorithm for matrix inversion using Kramer's rule
     ======================================================== */
  gettimeofday(&start_time, NULL);   // Record current sys time as start_time
  inverse( A, C, N);

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

   /* ====================================================
      Check if inverse is correct
      ==================================================== */
  int numErr = 0;
  double maxErr = 0;

  for (int i = 0; i < N; i++)
     for (int j = 0; j < N; j++)
     {
        // Compute i,j-th element of A * Ainv

        double sum = 0;

        for ( int k = 0; k < N; k++ )
           sum += A_org[i*N+k] * C[k*N+j];

//      printf("Mult[%d][%d] = %f\n", i, j, sum);

        if ( i == j )
        {
           if ( fabs(sum - 1.0) > Epsilon )  
           {
//            printf("+Err at matrix elem (%d,%d): %lf\n", i, j, fabs(sum-1.0));
              numErr++;
           }

           if ( fabs(sum - 1.0) > maxErr )
              maxErr = fabs(sum - 1.0);
        }
        else
        {
           if ( fabs(sum - 0.0) > Epsilon )  
           {
//            printf("+Err at matrix elem (%d,%d): %lf\n", i, j, fabs(sum-0.0));
              numErr++;
           }

           if ( fabs(sum - 0.0) > maxErr )
              maxErr = fabs(sum - 0.0);
        }
     }
   printf("Max err = %lf\n\n", maxErr);

   if ( numErr > 0 )
      printf("**** #errors = %d, inverse is NOT correct !\n", numErr);
   else
      printf("Inverse is correct !\n");


   // Free memory
   free(A);
   free(C);

  return 0;
}

