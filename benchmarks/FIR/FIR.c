#include <stdio.h>
#include <stdlib.h>


void FIR(int input[16], int output[16], int coeffs[5]) {
    int num_coeffs = 5;
    int num_samples = 16;
    for (int n = 0; n < num_samples; n++) {
        output[n] = 0.0;
        for (int k = 0; k < num_coeffs; k++) {
            if (n >= k) {
                output[n] += coeffs[k] * input[n - k];
            }
        }
    }
}

// int main() {
//     // Define filter coefficients (example: 5-tap filter)
//     double coeffs[] = {0.1, 0.15, 0.5, 0.15, 0.1};
//     int num_coeffs = sizeof(coeffs) / sizeof(coeffs[0]);

//     // Define input signal (example: 10 samples)
//     double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
//     int num_samples = sizeof(input) / sizeof(input[0]);

//     // Allocate memory for the output signal
//     double *output = (double*)malloc(num_samples * sizeof(double));

//     // Apply FIR filter
//     FIR(input, output, coeffs, num_coeffs, num_samples);

//     // Print the filtered output signal
//     printf("Filtered output:\n");
//     for (int i = 0; i < num_samples; i++) {
//         printf("output[%d] = %f\n", i, output[i]);
//     }

//     // Free allocated memory
//     free(output);

//     return 0;
// }
