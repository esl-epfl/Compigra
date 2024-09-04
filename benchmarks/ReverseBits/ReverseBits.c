#include <stdio.h>
#include <math.h>


unsigned ReverseBits ( unsigned *index_ptr, unsigned *NumBits_ptr, unsigned *rev_ptr )
{
    unsigned i, rev;

    unsigned index = *index_ptr;
    unsigned NumBits = *NumBits_ptr;

    #pragma CGRA
    for ( i=rev=0; i < NumBits; i++ )
    {
        rev = (rev << 1) | (index & 1);
        index >>= 1;
    }
    // *index_ptr = index;
    // *NumBits_ptr = NumBits;
    *rev_ptr = rev;

    return 0;
}


int main(){
  unsigned index = 325;
  unsigned NumBits = 32;
  unsigned result;
  unsigned rev = ReverseBits(index, NumBits, result);
  // printf("Index: ");
  // for (int i = sizeof(index) * 8 - 1; i >= 0; i--) {
  //   putchar((index & (1 << i)) ? '1' : '0');
  // }
  // printf("\nReverse: ");
  // for (int i = sizeof(rev) * 8 - 1; i >= 0; i--) {
  //   putchar((rev & (1 << i)) ? '1' : '0');
  // }

  // printf("\nReverse of %d is %d\n", index, rev);
  return 0;
  
}
