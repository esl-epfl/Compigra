#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define TEST_LENGTH 10

// uint32_t isqrt(uint32_t* in_ptr, uint32_t *result_ptr) 
// {
// 	uint32_t mask = 1<<14;
	
// 	uint32_t temp = 0;
// 	uint32_t result = *result_ptr;

// 	if(*in_ptr < 0) return 0;
// 	while(mask)
// 	{
// 		temp = result | mask;
// 		if((((uint32_t)temp)*((uint32_t)temp)) <= ((uint32_t)(*in_ptr)))
// 			result = temp;
// 		mask >>= 1;
// 		printf("flga: %d\n", ((uint32_t)temp)*((uint32_t)temp))-((uint32_t)(*in_ptr));
// 		printf("result: %d\n", result);
// 		printf("mask: %d\n\n", mask);
// 	}
// 	*result_ptr = result;
// 	return 0;
// }

int isqrt(unsigned* in_ptr, int *result_ptr) 
{
	int mask = 1<<14;
	
	int temp = 0;
	int result = *result_ptr;

	// if(*in_ptr < 0) return 0;
	while(mask)
	{
		temp = result | mask;
		if((((int)temp)*((int)temp)) <= ((int)(*in_ptr)))
			result = temp;
		mask >>= 1;
		// printf("flag: %d\n", ((int)temp)*((int)temp))-((int)(*in_ptr));
		// printf("result: %d\n", result);
		// printf("mask: %d\n\n", mask);
	}
	*result_ptr = result;
	return 0;
}


int main(int argc, char** argv) {
	int mathSqrt, mySqrt;
	int errors = 0;

	srand(time(NULL));
	int test_val = rand()%32768;
	int result = 0;
	// printf("test_val: %d\n", test_val);
	isqrt(&test_val, &result);

/*	for (int8_t i = 0; i < TEST_LENGTH; ++i) {
		uint32_t test_val = rand()%32768;
		mathSqrt = sqrt((double)test_val);
		mySqrt   = isqrt32(&test_val);

		if (mathSqrt != mySqrt) {
			printf("%d: sqrt(%d) = %d != %d\n", i, test_val, mathSqrt, mySqrt);
			errors++;
		} else {
			printf("%d: sqrt(%d) = %d = %d\n", i, test_val, mathSqrt, mySqrt);
		}
	}

	if(errors==0) {
		printf("SUCCESS: no error!\n");
		return EXIT_SUCCESS;
	} else {
		printf("FAILURE: %d errors out of %d tested values!\n", errors, TEST_LENGTH);
		return EXIT_FAILURE;
	}
*/
	return EXIT_FAILURE;
}
