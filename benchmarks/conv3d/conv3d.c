#include <stdio.h>
#include <stdlib.h>

#define NI 3  
#define NJ 15 
#define NK 15  
#define ND 2   
#define NM 5   
#define NL 5  

void conv3d(float input[NI][NJ][NK],
            float kernel[ND][NM][NL],
            float output[NI-ND+1][NJ-NM+1][NK-NL+1]) { 

    int input_depth = NI;
    int input_height = NJ;
    int input_width = NK;
    int kernel_depth = ND;
    int kernel_height = NM;
    int kernel_width = NL;

    int i, j, k, l, m, n, p;
    
    // Convolution operation
    for (i = 0; i < input_depth - kernel_depth + 1; i++) {
        for (j = 0; j < input_height - kernel_height + 1; j++) {
            for (k = 0; k < input_width - kernel_width + 1; k++) {
                output[i][j][k] = 0;
                for (l = 0; l < kernel_depth; l++) {
                    for (m = 0; m < kernel_height; m++) {
                        for (n = 0; n < kernel_width; n++) {
                            output[i][j][k] += input[i + l][j + m][k + n] * kernel[l][m][n];
                        }
                    }
                }
            }
        }
    }
}

