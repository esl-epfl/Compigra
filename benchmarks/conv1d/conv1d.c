#include <stdio.h>
#include <stdlib.h>

#define NI 64
#define NK 3

void conv1d(int in_size_ptr[1], 
            int kernel_size_ptr[1], 
            float input[NI], float kernel[NK], float output[NI-NK+1]) {
    int in_size = in_size_ptr[0];
    int kernel_size = kernel_size_ptr[0];

    int i, j;
    for (i = 0; i < in_size - kernel_size + 1; i++) {
        output[i] = 0;
        for (j = 0; j < kernel_size; j++) {
            output[i] += input[i + j] * kernel[j];
        }
    }
}
