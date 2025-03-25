/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

/* Problem size */
#define M 10
#define N 10

/* Main computational kernel. */
void symm(int alpha_ptr[1],
		 int beta_ptr[1],
		 int C[M][N],
		 int A[M][M],
		 int B[M][N])
{
  int i, j, k;
  int temp2;
  int m = M;
  int n = N;  

  int beta = beta_ptr[0];
  int alpha = alpha_ptr[0];

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      temp2 = 0;
      for (k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
  }
}

// /* Array initialization. */
// void init_array(int m, int n,
// 		int *alpha,
// 		int *beta,
// 		int C[M][N],
// 		int A[M][M],
// 		int B[M][N])
// {
//   int i, j;

//   *alpha = 1.5;
//   *beta = 1.2;
  
//   for (i = 0; i < m; i++) {
//     for (j = 0; j < n; j++) {
//       C[i][j] = (i + j) % 100 / m;
//       B[i][j] = (n + i - j) % 100 / m;
//     }
//   }

//   for (i = 0; i < m; i++) {
//     for (j = 0; j <= i; j++) {
//       A[i][j] = (i + j) % 100 / m;
//     }
//     for (j = i + 1; j < m; j++) {
//       A[i][j] = -999; // regions of arrays that should not be used
//     }
//   }
// }

// /* Print array for debugging or result verification */
// void print_array(int m, int n, int C[M][N]) {
//   int i, j;
//   for (i = 0; i < m; i++) {
//     for (j = 0; j < n; j++) {
//       printf("%d ", C[i][j]);
//       if ((i * n + j) % 20 == 0) printf("\n");
//     }
//   }
//   printf("\n");
// }

// int main(int argc, char **argv)
// {
//   /* Retrieve problem size. */
//   int m = M;
//   int n = N;

//   /* Variable declaration/allocation. */
//   int alpha;
//   int beta;
//   int C[M][N];
//   int A[M][M];
//   int B[M][N];

//   /* Initialize array(s). */
//   init_array(m, n, &alpha, &beta, C, A, B);

//   /* Run kernel. */
//   symm(m, n, alpha, beta, C, A, B);

//   /* Prevent dead-code elimination by printing the result. */
//   print_array(m, n, C);

//   return 0;
// }
