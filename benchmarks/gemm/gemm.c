/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <stdlib.h>

#define NI 10
#define NJ 10
#define NK 10

#define POLYBENCH_2D(var, dim1, dim2, ddim1, ddim2) var[ddim1][ddim2]

void gemm(int alpha_ptr[1],
          int beta_ptr[1],
          int C[NI][NJ],
          int A[NI][NK],
          int B[NK][NJ])
{
    int i, j, k;
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int alpha = alpha_ptr[0];
    int beta = beta_ptr[0];

    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++)
            C[i][j] *= beta;
        for (k = 0; k < nk; k++) {
            for (j = 0; j < nj; j++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }
    }
}

// void init_array(int ni, int nj, int nk,
//                 int *alpha,
//                 int *beta,
//                 int POLYBENCH_2D(C, NI, NJ, ni, nj),
//                 int POLYBENCH_2D(A, NI, NK, ni, nk),
//                 int POLYBENCH_2D(B, NK, NJ, nk, nj))
// {
//     int i, j;

//     *alpha = 2;
//     *beta = 3;

//     for (i = 0; i < ni; i++)
//         for (j = 0; j < nj; j++)
//             C[i][j] = (i * j + 1) % ni;
//     for (i = 0; i < ni; i++)
//         for (j = 0; j < nk; j++)
//             A[i][j] = (i * (j + 1) % nk);
//     for (i = 0; i < nk; i++)
//         for (j = 0; j < nj; j++)
//             B[i][j] = (i * (j + 2) % nj);
// }

// int main()
// {
//     int ni = NI;
//     int nj = NJ;
//     int nk = NK;

//     int alpha;
//     int beta;

//     int (*C)[NJ] = malloc(NI * sizeof(int[NJ]));
//     int (*A)[NK] = malloc(NI * sizeof(int[NK]));
//     int (*B)[NJ] = malloc(NK * sizeof(int[NJ]));

//     init_array(ni, nj, nk, &alpha, &beta, C, A, B);

//     clock_t start = clock();
//     gemm(ni, nj, nk, alpha, beta, C, A, B);
//     clock_t end = clock();

//     double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     printf("GEMM execution time: %f seconds\n", cpu_time_used);

//     // Free allocated memory
//     free(C);
//     free(A);
//     free(B);

//     return 0;
// }