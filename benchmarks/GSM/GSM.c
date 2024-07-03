#include<stdio.h>

# define GSM_ABS(a) ((a) < 0 ? ((a) == MIN_WORD ? MAX_WORD : -(a)) : (a))
#define MIN_WORD    ((-32767)-1)
#define MAX_WORD    ( 32767)

int GSM(int *dmax_ptr, int *temp_ptr, int *d){

    int dmax = *dmax_ptr;
    int temp = 0;

    #pragma CGRA
    for (int k = 0; k <= 39; k++) {
        temp = d[k];
        temp = GSM_ABS( temp );
        if (temp > dmax) dmax = temp;
    }

    *dmax_ptr = dmax;
    *temp_ptr = temp;

    return 0;
}

int main(){

	int d[50];
	GSM(0,0,d);


}
