#include<stdio.h>
/* SHA f()-functions */

#define f1(x,y,z)   ((x & y) | (~x & z))
#define f2(x,y,z)   (x ^ y ^ z)
#define f3(x,y,z)   ((x & y) | (x & z) | (y & z))
#define f4(x,y,z)   (x ^ y ^ z)

/* SHA constants */

#define CONST1      0x5a827999L
#define CONST2      0x6ed9eba1L
#define CONST3      0x8f1bbcdcL
#define CONST4      0xca62c1d6L

/* 32-bit rotate */

#define ROT32(x,n)  ((x << n) | (x >> (32 - n)))

#define FUNC(n,i)                       \
    temp = ROT32(A,5) + f##n(B,C,D) + E + W[i] + CONST##n;  \
    E = D; D = C; C = ROT32(B,30); B = A; A = temp



int SHA2(int W[20], int *result_ptr){
    int temp=0;
    int A=0;
    int B=0;
    int C=0;
    int D=0;
    int E=0;


    for (int i = 0; i < 20; ++i) {
        FUNC(1,i);
        // printf("A: %d, B: %d, C: %d, D: %d, E: %d\n", A, B, C, D, E);
    }

    *result_ptr = temp+A+B+C+D+E;
    return 0;

}

int main(){

	int W[80] = {1,1,1,1,1,1,1,1,1,1,
				2,2,2,2,2,2,2,2,2,2};
    int result;
	SHA2(W, &result);
    // printf("%d\n", result);
    return 0;


}
