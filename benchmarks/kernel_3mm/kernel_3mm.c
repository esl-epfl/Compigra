/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* 3mm.c: this file is part of PolyBench/C */

#include <stdio.h>

#define NI 4
#define NJ 4
#define NK 4
#define NL 4
#define NM 4

void kernel_3mm(int A[NI][NK],
                int B[NK][NJ],
                int C[NJ][NM],
                int D[NM][NL],
                int E[NI][NJ],
                int F[NJ][NL],
                int G[NI][NL])
{
    int i, j, k;
    int ni = NI;
    int nj = NJ;
    int nk = NK;
    int nl = NL; 
    int nm = NM;

    /* E := A*B */
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            E[i][j] = 0;
            for (k = 0; k < nk; ++k)
                E[i][j] += A[i][k] * B[k][j];
        }

    /* F := C*D */
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
        {
            F[i][j] = 0;
            for (k = 0; k < nm; ++k)
                F[i][j] += C[i][k] * D[k][j];
        }

    /* G := E*F */
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
        {
            G[i][j] = 0;
            for (k = 0; k < nj; ++k)
                G[i][j] += E[i][k] * F[k][j];
        }
}
