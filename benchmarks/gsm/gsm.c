// #include<stdio.h>

# define GSM_ABS(a) ((a) < 0 ? ((a) == MIN_WORD ? MAX_WORD : -(a)) : (a))
#define MIN_WORD    ((-32767)-1)
#define MAX_WORD    ( 32767)

int gsm(int dmax_ptr[1], int input_arr[50]){

    int dmax = 0;
    int temp = 0;

    for (int k = 0; k <= 39; k++) {
        temp = input_arr[k];
        temp = GSM_ABS( temp );
        if (temp > dmax) dmax = temp;
    }

    // *dmax_ptr = dmax;
    dmax_ptr[0] = dmax;


    return 0;
}

int main(){

	int d[50] = {-232, 456, 543, 3, 8, 9, 7, 67};
    int dmax = 0;
	gsm(&dmax,d);


}
