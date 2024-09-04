// Function to compute the dot product of two vectors
int DotProduct(int* x, int* y, int *n_ptr, int *result_ptr) {
    int result = 0;
    int n = *n_ptr;
    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    *result_ptr = result;
    return result;
}