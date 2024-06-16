#include <stdio.h>
#include <math.h>


int BitCount(long *x, int *n)
{
        // int n = 0;
/*
** The loop will execute once for each bit of x set, this is in average
** twice as fast as the shift/test method.
*/
        // if (x == 0){
        //     return 0;
        // }
        #pragma cgra
        
        do{
          *n = *n + 1;
        }while (0 != (*x = *x&(*x-1)));

        return(0);
}


int main(){

  int a = 123123;

  int res = BitCount(a,0);

  // printf("%d", res);
  return 0;
  
}
