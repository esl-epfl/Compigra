#include <stdio.h>
#include <math.h>


// int BitCount(int x)
// {
//         int n = 0;
// /*
// ** The loop will execute once for each bit of x set, this is in average
// ** twice as fast as the shift/test method.
// */
//         // if (x == 0){
//         //     return 0;
//         // }
//         #pragma cgra
        
//         do{
//           n = n + 1;
//         }while (0 != (x = x&(x-1)));

//         return(n);
// }

int BitCount(int *x, int *n)
{
  int x_val = *x;
  int n_val = *n;

  do{
    n_val = n_val + 1;
  }while (0 != (x_val = x_val&(x_val-1)));

  *x = x_val;
  *n = n_val;

  return(0);
}


int main(){

  int a = 123123;
  int n = 0;

  int res = BitCount(&a, &n);

  // printf("%d\n", n);
  return 0;
  
}
