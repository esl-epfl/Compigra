/* NIST Secure Hash Algorithm */
/* heavily modified by Uwe Hollerbach uh@alumni.caltech edu */
/* from Peter C. Gutmann's implementation as found in */
/* Applied Cryptography by Bruce Schneier */

/* NIST's proposed modification to SHA of 7/11/94 may be */
/* activated by defining USE_MODIFIED_SHA */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include "sha.h"

/* SHA f()-functions */
int temp, A, B, C, D, E;

#define f1(x,y,z)	((x & y) | (~x & z))
#define f2(x,y,z)	(x ^ y ^ z)
#define f3(x,y,z)	((x & y) | (x & z) | (y & z))
#define f4(x,y,z)	(x ^ y ^ z)

/* SHA constants */

#define CONST1		0x5a827999L
#define CONST2		0x6ed9eba1L
// #define CONST3		0x8f1bbcdcL
// #define CONST4		0xca62c1d6L

// the number is changed due to hardware representation
#define CONST3      0x4F1BBCDCL
#define CONST4      0x4A62C1D6L

/* 32-bit rotate */


#define ROT32(x,n)	((x << n) | (x >> (32 - n)))

#define FUNC(n,i)						\
    temp = ROT32(A,5) + f##n(B,C,D) + E + W[i] + CONST##n;	\
    E = D; D = C; C = ROT32(B,30); B = A; A = temp
    
/* do SHA transformation */

void sha_transform(int digest[5], int data[16],int W[80])
{
    int i;
    // int temp, A, B, C, D, E;


    for (i = 0; i < 16; ++i) {
	W[i] = data[i];
    }


    for (i = 16; i < 80; ++i) {
	W[i] = W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16];
    }
    A = digest[0];
    B = digest[1];
    C = digest[2];
    D = digest[3];
    E = digest[4];

    for (i = 0; i < 20; ++i) {
	FUNC(1,i);
    }

    for (i = 20; i < 40; ++i) {
	FUNC(2,i);
    }
    for (i = 40; i < 60; ++i) {
	FUNC(3,i);
    }
    for (i = 60; i < 80; ++i) {
	FUNC(4,i);
    }

    digest[0] += A;
    digest[1] += B;
    digest[2] += C;
    digest[3] += D;
    digest[4] += E;
}

/* initialize the SHA digest */

void sha_init(int digest[5], int *count_lo, int *count_hi)
{
    digest[0] = 0x67452301L;
    digest[1] = 0xefcdab89L;
    digest[2] = 0x98badcfeL;
    digest[3] = 0x10325476L;
    digest[4] = 0xc3d2e1f0L;
    *count_lo = 0L;
    *count_hi = 0L;
}

