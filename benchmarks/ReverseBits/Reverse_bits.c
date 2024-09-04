#include <stdio.h>
#include <math.h>


int ReverseBits ( int index, int NumBits )
{
    int i, rev;

    #pragma CGRA
    for ( i=rev=0; i < NumBits; i++ )
    {
        rev = (rev << 1) | (index & 1);
        index >>= 1;
        printf("%d, %d\n",index, rev);
    }

    return rev;
}


int main(){
  int index = 655162044;
  int NumBits = 32;
  int rev = ReverseBits(index, NumBits);
  printf("Index: ");
  for (int i = sizeof(index) * 8 - 1; i >= 0; i--) {
    putchar((index & (1 << i)) ? '1' : '0');
  }
  printf("\nReverse: ");
  for (int i = sizeof(rev) * 8 - 1; i >= 0; i--) {
    putchar((rev & (1 << i)) ? '1' : '0');
  }

  printf("\nReverse of %d is %d\n", index, rev);
  return 0;
  
}
