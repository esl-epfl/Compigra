#include <stdio.h>
#include <stdlib.h>

#define NI 12 
#define NJ 12  
#define NM 3
#define NK 3   

void conv2d(int input_size[2],      
            int kernel_size[2],     
            float input[NI][NJ],    
            float kernel[NM][NK],   
            float output[NI-NM+1][NI-NK+1]) { 

    int input_height = input_size[0];
    int input_width = input_size[1];
    int kernel_height = kernel_size[0];
    int kernel_width = kernel_size[1];

    int i, j, m, n;
    
    // Convolution operation
    for (i = 0; i < input_height - kernel_height + 1; i++) {
        for (j = 0; j < input_width - kernel_width + 1; j++) {
            output[i][j] = 0;
            for (m = 0; m < kernel_height; m++) {
                for (n = 0; n < kernel_width; n++) {
                    output[i][j] += input[i + m][j + n] * kernel[m][n];
                }
            }
        }
    }
}
